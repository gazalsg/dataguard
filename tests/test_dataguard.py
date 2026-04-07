"""
DataGuard — Tests
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import pandas as pd
import pytest

from dataset_gen import generate_easy, generate_medium, generate_hard
from grader import grade
from env import DataGuardEnv
from models import ActionType, DataGuardAction, FillStrategy, TargetDtype


# ---------------------------------------------------------------------------
# Dataset generator tests
# ---------------------------------------------------------------------------

class TestDatasetGenerator:
    def test_easy_shape(self):
        df, schema = generate_easy()
        assert df.shape == (10, 4)
        assert "name" in df.columns
        assert "signup_date" in df.columns

    def test_easy_names_are_uppercase(self):
        df, _ = generate_easy()
        for name in df["name"]:
            assert name == name.upper(), f"Expected uppercase, got: {name!r}"

    def test_easy_dates_are_wrong_format(self):
        df, _ = generate_easy()
        for d in df["signup_date"]:
            assert "/" in str(d), f"Expected MM/DD/YYYY, got: {d!r}"

    def test_medium_has_duplicates(self):
        df, _ = generate_medium()
        assert df.duplicated().sum() > 0

    def test_medium_has_null_emails(self):
        df, _ = generate_medium()
        assert df["email"].isna().sum() > 0

    def test_medium_age_is_float(self):
        df, _ = generate_medium()
        assert df["age"].dtype == float

    def test_hard_shape(self):
        df, _, rate = generate_hard()
        assert len(df) == 500
        assert isinstance(rate, float)
        assert 1.0 < rate < 2.0

    def test_hard_has_gbp_prices(self):
        df, _, _ = generate_hard()
        gbp_count = df["price"].astype(str).str.contains("£").sum()
        assert gbp_count > 0


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGrader:
    def test_easy_zero_score_on_raw(self):
        df, _ = generate_easy()
        result = grade("easy", df, len(df))
        assert result.total < 0.5, "Raw dataset should score below 0.5"

    def test_easy_perfect_score_when_solved(self):
        df, _ = generate_easy()
        df["name"] = df["name"].str.title()
        df["signup_date"] = pd.to_datetime(df["signup_date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
        result = grade("easy", df, len(df))
        assert result.total == 1.0, f"Expected 1.0, got {result.total}"

    def test_medium_zero_when_raw(self):
        df, _ = generate_medium()
        result = grade("medium", df, len(df))
        # Raw dataset has duplicates and float ages, so can't score perfectly
        assert result.total < 0.99
        assert result.breakdown.dtype_fixes == 0.0  # age is still float

    def test_medium_full_score_when_solved(self):
        df, _ = generate_medium()
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.dropna(subset=["email"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["age"] = pd.to_numeric(df["age"], errors="coerce").round().astype("Int64")
        result = grade("medium", df, 100)
        assert result.total >= 0.95

    def test_hard_retention_penalty(self):
        df, _, _ = generate_hard()
        original = len(df)
        # Drop 20% of rows — should trigger penalty
        df = df.head(int(original * 0.75)).copy()
        result = grade("hard", df, original)
        assert result.breakdown.row_retention_penalty < 0.0

    def test_unknown_task_raises(self):
        df, _ = generate_easy()
        with pytest.raises(ValueError, match="Unknown task"):
            grade("banana", df, len(df))


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_reset_returns_observation(self):
        env = DataGuardEnv("easy")
        result = env.reset()
        assert result.observation.task_name == "easy"
        assert result.observation.step == 0
        assert result.done is False

    def test_step_increments_step_count(self):
        env = DataGuardEnv("easy")
        env.reset()
        action = DataGuardAction(
            action=ActionType.STANDARDIZE_FORMAT,
            column="name",
            format="Title Case",
        )
        result = env.step(action)
        assert result.observation.step == 1

    def test_validate_schema_ends_episode(self):
        env = DataGuardEnv("easy")
        env.reset()
        # Solve it first
        env.step(DataGuardAction(action=ActionType.STANDARDIZE_FORMAT, column="name", format="Title Case"))
        env.step(DataGuardAction(action=ActionType.STANDARDIZE_FORMAT, column="signup_date", format="YYYY-MM-DD"))
        result = env.step(DataGuardAction(action=ActionType.VALIDATE_SCHEMA))
        assert result.done is True
        assert result.observation.reward_so_far == 1.0

    def test_step_after_done_raises(self):
        env = DataGuardEnv("easy")
        env.reset()
        env.step(DataGuardAction(action=ActionType.VALIDATE_SCHEMA))
        with pytest.raises(RuntimeError, match="Episode is done"):
            env.step(DataGuardAction(action=ActionType.VALIDATE_SCHEMA))

    def test_medium_drop_duplicates(self):
        env = DataGuardEnv("medium")
        env.reset()
        before_rows = env._df.shape[0]
        env.step(DataGuardAction(action=ActionType.DROP_DUPLICATES))
        after_rows = env._df.shape[0]
        assert after_rows < before_rows

    def test_state_returns_dict(self):
        env = DataGuardEnv("hard")
        env.reset()
        s = env.state()
        assert "task" in s
        assert s["task"] == "hard"
        assert "gbp_usd_rate" in s
        assert isinstance(s["gbp_usd_rate"], float)

    def test_hard_convert_units(self):
        env = DataGuardEnv("hard")
        env.reset()
        rate = env._gbp_usd_rate
        env.step(DataGuardAction(
            action=ActionType.CONVERT_UNITS,
            column="price",
            from_unit="GBP",
            to_unit="USD",
            rate=rate,
        ))
        # After conversion all non-null values should be numeric
        env.step(DataGuardAction(action=ActionType.FIX_DTYPE, column="price", target_type=TargetDtype.FLOAT))
        non_numeric = env._df["price"].apply(
            lambda x: not isinstance(x, (int, float)) if x is not None else False
        ).sum()
        assert non_numeric == 0


# ---------------------------------------------------------------------------
# Reward shaping tests
# ---------------------------------------------------------------------------

class TestRewardShaping:
    def test_reward_clamped_to_one(self):
        df, _ = generate_easy()
        df["name"] = df["name"].str.title()
        df["signup_date"] = pd.to_datetime(df["signup_date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
        r = grade("easy", df, len(df))
        assert 0.0 <= r.total <= 1.0

    def test_partial_reward_for_partial_fix(self):
        df, _ = generate_easy()
        # Fix only names, not dates
        df["name"] = df["name"].str.title()
        r = grade("easy", df, len(df))
        assert 0.0 < r.total < 1.0, "Should get partial credit for partial fix"