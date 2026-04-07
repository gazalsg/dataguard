"""
DataGuard — Task: Easy
10 rows | Fix name casing + date format
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd

from dataset_gen import generate_easy

TASK_NAME    = "easy"
MAX_STEPS    = 6
REWARD_MAX   = 1.0
DIFFICULTY   = "easy"

DESCRIPTION  = """
The agent receives a 10-row customer table with two issues:
  1. The 'name' column is stored in ALL CAPS — it must be Title Case.
  2. The 'signup_date' column uses MM/DD/YYYY — it must be ISO 8601 (YYYY-MM-DD).

The agent should call standardize_format twice (once per column) then validate_schema.
An optimal agent solves this in 3 steps and scores 1.0.
"""

SCHEMA_REQUIREMENTS: Dict[str, Any] = {
    "name":        {"format": "Title Case",  "description": "Each word capitalised, e.g. 'Alice Johnson'"},
    "signup_date": {"format": "YYYY-MM-DD",  "description": "ISO 8601 date string"},
}

REWARD_BREAKDOWN = {
    "name_title_case":  0.5,
    "date_iso_format":  0.5,
}


def load(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return (dirty_df, schema_requirements) for this task."""
    df, schema = generate_easy(seed)
    return df, schema