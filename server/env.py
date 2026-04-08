"""
DataGuard — Core Environment
=============================
This is the heart of the environment. It manages episode state, dispatches
agent actions against the live DataFrame, and builds the observation the
agent sees at each step.

The design philosophy here is intentional constraint: the agent can only
call actions from a fixed toolkit with enumerated arguments. No free-form
pandas expressions, no arbitrary code execution. This makes the grader
fully deterministic and removes the possibility of reward hacking through
clever string manipulation.

One environment instance per task is held in memory by the FastAPI server.
State is reset on every reset() call — nothing persists between episodes.
"""

from __future__ import annotations

import re
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from server.dataset_gen import generate_easy, generate_medium, generate_hard
from server.grader import grade
from server.models import (
    ActionType,
    ColumnSummary,
    DataGuardAction,
    DataGuardObservation,
    DataGuardReward,
    FillStrategy,
    TargetDtype,
)


# ---------------------------------------------------------------------------
# Step result container
# ---------------------------------------------------------------------------

class StepResult:
    def __init__(
        self,
        observation: DataGuardObservation,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ):
        self.observation = observation
        self.reward      = reward
        self.done        = done
        self.info        = info


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

MAX_STEPS = {
    "easy":   6,
    "medium": 10,
    "hard":   18,
}

AVAILABLE_ACTIONS = [
    'fix_dtype(column, target_type)        — Cast column dtype [int|float|string|date]',
    'standardize_format(column, format)    — Apply format rule (e.g. YYYY-MM-DD, Title Case)',
    'drop_duplicates(subset)               — Remove duplicate rows by subset of columns',
    'fill_nulls(column, strategy)          — Fill nulls [mean|median|mode|drop]',
    'convert_units(column, from, to, rate) — Convert numeric values between units',
    'drop_rows(condition)                  — Drop rows matching a pandas query condition',
    'validate_schema()                     — Score current state and end episode',
]


class DataGuardEnv:
    """
    OpenEnv-compatible environment for automated data sanitisation.

    Lifecycle
    ---------
    env = DataGuardEnv(task="easy")
    result = env.reset()
    while not result.done:
        action = agent.act(result.observation)
        result = env.step(action)
    """

    def __init__(self, task: str = "easy", seed: int = 42):
        if task not in ("easy", "medium", "hard"):
            raise ValueError(f"task must be 'easy', 'medium', or 'hard', got {task!r}")
        self.task = task
        self.seed = seed
        self._df:               Optional[pd.DataFrame]  = None
        self._schema:           Optional[Dict[str, Any]] = None
        self._original_df:      Optional[pd.DataFrame]  = None
        self._gbp_usd_rate:     Optional[float]         = None
        self._step_count:       int                     = 0
        self._cumulative_reward: float                  = 0.0
        self._done:             bool                    = False
        self._last_action_result: Optional[str]         = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> StepResult:
        """Start a fresh episode. Returns initial observation."""
        self._step_count        = 0
        self._cumulative_reward = 0.0
        self._done              = False
        self._last_action_result = "Episode started. Inspect the dataset and begin cleaning."

        if self.task == "easy":
            self._df, self._schema = generate_easy(self.seed)
        elif self.task == "medium":
            self._df, self._schema = generate_medium(self.seed)
        else:  # hard
            self._df, self._schema, self._gbp_usd_rate = generate_hard(self.seed)

        self._original_df = self._df.copy(deep=True)

        obs = self._build_observation()
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    def step(self, action: DataGuardAction) -> StepResult:
        """Execute one action and return next observation + reward delta."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        reward_delta = 0.0
        done         = False

        try:
            reward_delta, done = self._execute_action(action)
        except Exception as exc:
            self._last_action_result = f"ERROR: {exc}"

        self._cumulative_reward = round(self._cumulative_reward + reward_delta, 4)

        # Force end if max steps reached
        if self._step_count >= MAX_STEPS[self.task] and not done:
            # Give partial score even if agent never called validate_schema
            final_reward = grade(self.task, self._df, len(self._original_df))
            reward_delta += final_reward.total - self._cumulative_reward
            self._cumulative_reward = final_reward.total
            done = True
            self._last_action_result = (
                f"Max steps reached. Auto-graded: {final_reward.message}"
            )

        self._done = done
        obs = self._build_observation()
        return StepResult(
            observation=obs,
            reward=reward_delta,
            done=done,
            info={"cumulative_reward": self._cumulative_reward},
        )

    def state(self) -> Dict[str, Any]:
        """Return raw environment state (useful for debugging / serialisation)."""
        return {
            "task":               self.task,
            "step":               self._step_count,
            "done":               self._done,
            "cumulative_reward":  self._cumulative_reward,
            "df_shape":           list(self._df.shape) if self._df is not None else None,
            "schema":             self._schema,
            "gbp_usd_rate":       self._gbp_usd_rate,
        }

    # ------------------------------------------------------------------
    # Action executor
    # ------------------------------------------------------------------

    def _execute_action(self, action: DataGuardAction) -> Tuple[float, bool]:
        """Dispatch action and return (reward_delta, done)."""
        df = self._df

        # ── validate_schema ───────────────────────────────────────────
        if action.action == ActionType.VALIDATE_SCHEMA:
            result = grade(self.task, df, len(self._original_df))
            self._last_action_result = f"GRADED: {result.message}"
            # Return the *incremental* reward (total minus what we've given already)
            delta = max(0.0, result.total - self._cumulative_reward)
            return round(delta, 4), True

        # ── fix_dtype ─────────────────────────────────────────────────
        if action.action == ActionType.FIX_DTYPE:
            col, ttype = self._require(action.column, action.target_type, "column and target_type")
            before_nulls = df[col].isna().sum()

            if ttype == TargetDtype.INT:
                df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
            elif ttype == TargetDtype.FLOAT:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif ttype == TargetDtype.STRING:
                df[col] = df[col].astype(str)
            elif ttype == TargetDtype.DATE:
                df[col] = pd.to_datetime(df[col], infer_format=True, errors="coerce").dt.strftime("%Y-%m-%d")

            after_nulls = df[col].isna().sum()
            self._last_action_result = (
                f"fix_dtype({col!r}, {ttype}) — "
                f"nulls before={before_nulls} after={after_nulls}"
            )
            return 0.0, False  # reward only on validate_schema

        # ── standardize_format ────────────────────────────────────────
        if action.action == ActionType.STANDARDIZE_FORMAT:
            col, fmt = self._require(action.column, action.format, "column and format")

            if fmt in ("YYYY-MM-DD", "ISO8601", "iso8601"):
                df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
                self._last_action_result = f"standardize_format({col!r}) → ISO 8601"
            elif fmt.lower() in ("title case", "title_case", "titlecase"):
                df[col] = df[col].astype(str).str.title()
                self._last_action_result = f"standardize_format({col!r}) → Title Case"
            elif fmt.lower() in ("uppercase", "upper"):
                df[col] = df[col].astype(str).str.upper()
                self._last_action_result = f"standardize_format({col!r}) → UPPERCASE"
            elif fmt.lower() in ("lowercase", "lower"):
                df[col] = df[col].astype(str).str.lower()
                self._last_action_result = f"standardize_format({col!r}) → lowercase"
            else:
                raise ValueError(
                    f"Unsupported format {fmt!r}. Use: 'YYYY-MM-DD', 'Title Case', 'uppercase', 'lowercase'."
                )
            return 0.0, False

        # ── drop_duplicates ───────────────────────────────────────────
        if action.action == ActionType.DROP_DUPLICATES:
            before = len(df)
            subset = action.subset or None
            df.drop_duplicates(subset=subset, inplace=True)
            df.reset_index(drop=True, inplace=True)
            dropped = before - len(df)
            self._last_action_result = f"drop_duplicates({subset}) — removed {dropped} rows"
            return 0.0, False

        # ── fill_nulls ────────────────────────────────────────────────
        if action.action == ActionType.FILL_NULLS:
            col, strategy = self._require(action.column, action.strategy, "column and strategy")
            null_count = df[col].isna().sum()

            if strategy == FillStrategy.DROP:
                df.dropna(subset=[col], inplace=True)
                df.reset_index(drop=True, inplace=True)
                self._last_action_result = f"fill_nulls({col!r}, drop) — dropped {null_count} rows"
            elif strategy == FillStrategy.MEAN:
                fill_val = pd.to_numeric(df[col], errors="coerce").mean()
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_val)
                self._last_action_result = f"fill_nulls({col!r}, mean={fill_val:.2f})"
            elif strategy == FillStrategy.MEDIAN:
                fill_val = pd.to_numeric(df[col], errors="coerce").median()
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_val)
                self._last_action_result = f"fill_nulls({col!r}, median={fill_val:.2f})"
            elif strategy == FillStrategy.MODE:
                fill_val = df[col].mode()[0] if not df[col].mode().empty else ""
                df[col] = df[col].fillna(fill_val)
                self._last_action_result = f"fill_nulls({col!r}, mode={fill_val!r})"
            return 0.0, False

        # ── convert_units ─────────────────────────────────────────────
        if action.action == ActionType.CONVERT_UNITS:
            col  = self._require_one(action.column,    "column")
            rate = self._require_one(action.rate,      "rate")

            def _parse_and_convert(val: Any) -> Optional[float]:
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return round(float(val) * rate, 4)
                s = str(val).strip()
                # Strip currency symbols
                s_clean = re.sub(r"[£$€¥₹]", "", s).strip()
                try:
                    return round(float(s_clean) * rate, 4)
                except ValueError:
                    return None

            df[col] = df[col].apply(_parse_and_convert)
            self._last_action_result = (
                f"convert_units({col!r}, rate={rate}) — "
                f"nulls after={df[col].isna().sum()}"
            )
            return 0.0, False

        # ── drop_rows ─────────────────────────────────────────────────
        if action.action == ActionType.DROP_ROWS:
            condition = self._require_one(action.condition, "condition")
            before = len(df)
            try:
                mask = df.query(condition)
                df.drop(index=mask.index, inplace=True)
                df.reset_index(drop=True, inplace=True)
                dropped = before - len(df)
                self._last_action_result = f"drop_rows({condition!r}) — removed {dropped} rows"
            except Exception as exc:
                raise ValueError(f"Invalid query condition: {exc}")
            return 0.0, False

        raise ValueError(f"Unknown action type: {action.action!r}")

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> DataGuardObservation:
        df = self._df
        cols = []
        for col in df.columns:
            sample = df[col].dropna().head(5).tolist()
            cols.append(ColumnSummary(
                name=col,
                dtype=str(df[col].dtype),
                null_count=int(df[col].isna().sum()),
                sample=sample,
            ))

        schema = deepcopy(self._schema) or {}
        if self._gbp_usd_rate is not None:
            schema["_exchange_rate_hint"] = {
                "GBP_to_USD": self._gbp_usd_rate,
                "note": "Use this rate when calling convert_units on the price column",
            }

        return DataGuardObservation(
            task_name           = self.task,
            step                = self._step_count,
            total_rows          = len(df),
            total_cols          = len(df.columns),
            original_row_count  = len(self._original_df) if self._original_df is not None else len(df),
            columns             = cols,
            schema_requirements = schema,
            available_actions   = AVAILABLE_ACTIONS,
            reward_so_far       = self._cumulative_reward,
            last_action_result  = self._last_action_result,
            hint                = self._get_hint(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_hint(self) -> Optional[str]:
        """Inject a gentle hint at specific steps to guide stuck agents."""
        if self._step_count == 0:
            return "Inspect schema_requirements carefully before acting."
        if self._step_count == 1 and self.task == "hard" and self._gbp_usd_rate:
            return (
                f"Hint: The 'price' column contains mixed currency strings (£ and $). "
                f"Use convert_units with rate={self._gbp_usd_rate} for GBP values, "
                f"then strip the $ symbol with fix_dtype(column='price', target_type='float')."
            )
        return None

    @staticmethod
    def _require(val1: Any, val2: Any, label: str) -> Tuple[Any, Any]:
        if val1 is None or val2 is None:
            raise ValueError(f"This action requires: {label}")
        return val1, val2

    @staticmethod
    def _require_one(val: Any, label: str) -> Any:
        if val is None:
            raise ValueError(f"This action requires: {label}")
        return val
