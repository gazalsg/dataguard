"""
DataGuard — Deterministic Grader
All scoring is done programmatically — no LLM judge.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np

from models import DataGuardReward, RewardBreakdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ID_RE    = re.compile(r"^[A-Z0-9]{8}$")


def _is_iso_date(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    try:
        pd.to_datetime(val, format="%Y-%m-%d")
        return True
    except Exception:
        return False


def _is_title_case(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    return val == val.title()


def _is_valid_email(val: Any) -> bool:
    if pd.isna(val):
        return False
    return bool(_EMAIL_RE.match(str(val)))


def _is_valid_id(val: Any) -> bool:
    return bool(_ID_RE.match(str(val)))


def _is_usd_float(val: Any) -> bool:
    """True if value is a plain float/int (already converted to USD)."""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and not pd.isna(val)


# ---------------------------------------------------------------------------
# Per-task graders
# ---------------------------------------------------------------------------

def grade_easy(df: pd.DataFrame, original_row_count: int) -> DataGuardReward:
    """
    Reward breakdown (total = 1.0):
      - name Title Case:       0.5
      - signup_date ISO 8601:  0.5
    No retention penalty for easy task.
    """
    bd = RewardBreakdown()

    # Name check
    if "name" in df.columns:
        frac_title = df["name"].apply(_is_title_case).mean()
        bd.format_fixes += round(0.5 * frac_title, 4)

    # Date check
    if "signup_date" in df.columns:
        frac_iso = df["signup_date"].apply(_is_iso_date).mean()
        bd.format_fixes += round(0.5 * frac_iso, 4)

    total = min(bd.format_fixes, 1.0)
    msg   = f"Name={bd.format_fixes:.2f} (combined with date score)"
    return DataGuardReward(total=total, breakdown=bd, message=msg)


def grade_medium(df: pd.DataFrame, original_row_count: int) -> DataGuardReward:
    """
    Reward breakdown (total = 1.0):
      - No duplicates:   0.33
      - No null emails:  0.34  (emails must also be valid format)
      - Age dtype int:   0.33
    """
    bd = RewardBreakdown()
    msgs = []

    # Duplicate check
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        bd.duplicate_removal = 0.33
        msgs.append("✓ No duplicates")
    else:
        bd.duplicate_removal = round(0.33 * max(0, 1 - dup_count / original_row_count), 4)
        msgs.append(f"✗ {dup_count} duplicates remain")

    # Email check
    if "email" in df.columns:
        frac_valid = df["email"].apply(_is_valid_email).mean()
        bd.null_handling = round(0.34 * frac_valid, 4)
        msgs.append(f"Email valid: {frac_valid:.0%}")

    # Age dtype check
    if "age" in df.columns:
        try:
            is_int = pd.api.types.is_integer_dtype(df["age"])
            in_range = ((df["age"] >= 0) & (df["age"] <= 120)).all() if is_int else False
            if is_int and in_range:
                bd.dtype_fixes = 0.33
                msgs.append("✓ Age is int in range")
            elif is_int:
                bd.dtype_fixes = 0.20
                msgs.append("✗ Age is int but out of range values exist")
            else:
                msgs.append("✗ Age is not int dtype")
        except Exception:
            msgs.append("✗ Age check failed")

    total = min(bd.dtype_fixes + bd.null_handling + bd.duplicate_removal, 1.0)
    return DataGuardReward(total=total, breakdown=bd, message=" | ".join(msgs))


def grade_hard(df: pd.DataFrame, original_row_count: int) -> DataGuardReward:
    """
    Reward breakdown (total = 1.0):
      - All prices USD float:  0.34
      - All dates ISO 8601:    0.33
      - All IDs valid:         0.33
      - Row retention penalty: up to -0.20 if >10% valid rows dropped
    """
    bd = RewardBreakdown()
    msgs = []

    # Price check
    if "price" in df.columns:
        frac_usd = df["price"].apply(_is_usd_float).mean()
        bd.unit_conversion = round(0.34 * frac_usd, 4)
        msgs.append(f"Price USD: {frac_usd:.0%}")

    # Date check
    if "event_date" in df.columns:
        frac_iso = df["event_date"].apply(_is_iso_date).mean()
        bd.format_fixes = round(0.33 * frac_iso, 4)
        msgs.append(f"Date ISO: {frac_iso:.0%}")

    # ID check
    if "record_id" in df.columns:
        frac_valid = df["record_id"].apply(_is_valid_id).mean()
        bd.schema_validation = round(0.33 * frac_valid, 4)
        msgs.append(f"ID valid: {frac_valid:.0%}")

    # Row retention penalty
    current_rows   = len(df)
    retention_rate = current_rows / original_row_count if original_row_count > 0 else 1.0
    if retention_rate < 0.90:
        drop_pct = 1.0 - retention_rate
        bd.row_retention_penalty = round(-min(drop_pct * 2, 0.20), 4)
        msgs.append(f"⚠ Retention penalty: {bd.row_retention_penalty:.2f} ({drop_pct:.0%} rows dropped)")

    raw_total = (
        bd.unit_conversion
        + bd.format_fixes
        + bd.schema_validation
        + bd.row_retention_penalty
    )
    total = round(max(0.0, min(raw_total, 1.0)), 4)
    return DataGuardReward(total=total, breakdown=bd, message=" | ".join(msgs))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


def grade(task_name: str, df: pd.DataFrame, original_row_count: int) -> DataGuardReward:
    """Entry point — call this from env.py."""
    grader = GRADERS.get(task_name)
    if grader is None:
        raise ValueError(f"Unknown task: {task_name!r}. Must be one of {list(GRADERS)}")
    return grader(df, original_row_count)


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset_gen import generate_easy, generate_medium, generate_hard

    # Easy — fully solved
    df_e, _ = generate_easy()
    df_e["name"]        = df_e["name"].str.title()
    df_e["signup_date"] = pd.to_datetime(df_e["signup_date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    r = grade("easy", df_e, len(df_e))
    print("Easy (solved):", r.total, r.message)

    # Medium — unsolved
    df_m, _ = generate_medium()
    r = grade("medium", df_m, len(df_m))
    print("Medium (raw):", r.total, r.message)

    # Hard — unsolved
    df_h, _, _ = generate_hard()
    r = grade("hard", df_h, len(df_h))
    print("Hard (raw):", r.total, r.message)