"""
DataGuard — Task: Medium
100 rows | Remove duplicates + fill null emails + fix age dtype
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd

from dataset_gen import generate_medium

TASK_NAME    = "medium"
MAX_STEPS    = 10
REWARD_MAX   = 1.0
DIFFICULTY   = "medium"

DESCRIPTION  = """
The agent receives a 100-row user table with three compounding issues:
  1. ~10% of rows are exact duplicates (must be removed).
  2. ~15% of emails are NULL (must be dropped or the rows removed — not filled with fake values).
  3. The 'age' column is stored as float64 (e.g. 34.7) — must be cast to int in range 0-120.

The agent must decide the right order of operations (drop dupes first, then handle nulls,
then fix dtype) and call validate_schema when done.
An optimal agent solves this in 4 steps and scores ~1.0.
"""

SCHEMA_REQUIREMENTS: Dict[str, Any] = {
    "email": {
        "format":   "valid_email",
        "nullable": False,
        "description": "No nulls; must match pattern user@domain.tld",
    },
    "age": {
        "dtype":       "int",
        "min":         0,
        "max":         120,
        "description": "Integer age in range 0-120",
    },
    "id": {
        "unique":      True,
        "description": "No duplicate rows (id is the natural key)",
    },
}

REWARD_BREAKDOWN = {
    "no_duplicates":  0.33,
    "no_null_emails": 0.34,
    "age_int_dtype":  0.33,
}


def load(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return (dirty_df, schema_requirements) for this task."""
    df, schema = generate_medium(seed)
    return df, schema