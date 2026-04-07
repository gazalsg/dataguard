"""
DataGuard — Synthetic Dataset Generator
Generates reproducible, seeded dirty datasets for each task difficulty.
"""

from __future__ import annotations

import random
import string
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_date(start: date, end: date, rng: np.random.Generator) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta)))


def _random_email(name: str) -> str:
    domains = ["gmail.com", "yahoo.com", "outlook.com", "example.org"]
    return f"{name.lower().replace(' ', '.')}@{random.choice(domains)}"


def _random_id(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


# ---------------------------------------------------------------------------
# Task 1 — EASY
# 10 rows | errors: wrong name case, wrong date format
# ---------------------------------------------------------------------------

def generate_easy(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns (dirty_df, schema_requirements).

    Errors injected:
      - 'name' column: all UPPERCASE instead of Title Case
      - 'signup_date' column: MM/DD/YYYY instead of ISO 8601 (YYYY-MM-DD)
    """
    rng = np.random.default_rng(seed)

    names = [
        "ALICE JOHNSON", "BOB SMITH", "CAROL WHITE", "DAVID BROWN",
        "EVA GREEN", "FRANK BLACK", "GRACE HALL", "HENRY FORD",
        "IRENE WOOD", "JACK STONE",
    ]

    start = date(2020, 1, 1)
    dates_iso = [_random_date(start, date(2024, 12, 31), rng) for _ in range(10)]
    # Inject wrong format
    dates_wrong = [d.strftime("%m/%d/%Y") for d in dates_iso]

    df = pd.DataFrame({
        "id":          range(1, 11),
        "name":        names,
        "signup_date": dates_wrong,
        "age":         rng.integers(18, 65, size=10).tolist(),
    })

    schema = {
        "name":        {"format": "Title Case",  "description": "Each word capitalised"},
        "signup_date": {"format": "YYYY-MM-DD",  "description": "ISO 8601 date string"},
    }

    return df, schema


# ---------------------------------------------------------------------------
# Task 2 — MEDIUM
# 100 rows | errors: duplicate rows, null emails, age stored as float
# ---------------------------------------------------------------------------

def generate_medium(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Errors injected:
      - 10% duplicate rows
      - ~15% null values in 'email'
      - 'age' stored as float64 (should be int, range 0-120)
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    n_base = 90
    first_names = ["Alice","Bob","Carol","David","Eva","Frank","Grace","Henry","Irene","Jack"]
    last_names  = ["Johnson","Smith","White","Brown","Green","Black","Hall","Ford","Wood","Stone"]

    names  = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_base)]
    emails = [_random_email(n) for n in names]
    ages   = rng.uniform(18.0, 80.0, size=n_base)  # floats — should be int

    df = pd.DataFrame({
        "id":    range(1, n_base + 1),
        "name":  names,
        "email": emails,
        "age":   ages,
    })

    # Inject nulls in email (~15%)
    null_idx = rng.choice(n_base, size=14, replace=False)
    df.loc[null_idx, "email"] = np.nan

    # Inject 10 duplicate rows
    dup_rows = df.sample(10, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    schema = {
        "email": {"format": "valid_email",    "nullable": False},
        "age":   {"dtype": "int",             "min": 0, "max": 120},
        "id":    {"unique": True,             "description": "No duplicate rows"},
    }

    return df, schema


# ---------------------------------------------------------------------------
# Task 3 — HARD
# 500 rows | errors: mixed currency, inconsistent dates, corrupted IDs
# Exchange rate injected into observation for agent to read & use
# ---------------------------------------------------------------------------

def generate_hard(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any], float]:
    """
    Returns (dirty_df, schema_requirements, gbp_to_usd_rate).

    Errors injected:
      - 'price': ~40% values are in GBP (prefixed with '£'), rest in USD ('$') or bare float
      - 'event_date': mix of YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY formats
      - 'record_id': ~20% are corrupted (wrong length / invalid chars)
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    n = 500
    gbp_to_usd = round(float(rng.uniform(1.20, 1.35)), 4)  # e.g. 1.2731

    # --- price column ---
    prices_usd = rng.uniform(10.0, 1000.0, size=n)
    price_col  = []
    for i, p in enumerate(prices_usd):
        r = rng.random()
        if r < 0.40:
            price_col.append(f"£{round(p / gbp_to_usd, 2)}")   # GBP string
        elif r < 0.70:
            price_col.append(f"${round(p, 2)}")                  # USD string
        else:
            price_col.append(round(p, 2))                        # bare float

    # --- event_date column ---
    start = date(2018, 1, 1)
    raw_dates = [_random_date(start, date(2024, 12, 31), rng) for _ in range(n)]
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]
    date_col = [d.strftime(random.choice(formats)) for d in raw_dates]

    # --- record_id column ---
    good_ids = [_random_id(8) for _ in range(n)]
    id_col   = list(good_ids)
    corrupt_idx = rng.choice(n, size=int(n * 0.20), replace=False)
    for i in corrupt_idx:
        bad_type = rng.integers(0, 3)
        if bad_type == 0:
            id_col[i] = _random_id(rng.integers(3, 6))          # wrong length
        elif bad_type == 1:
            id_col[i] = _random_id(8).lower()                   # lowercase (invalid)
        else:
            id_col[i] = "CORRUPT_" + _random_id(4)              # clearly bad

    df = pd.DataFrame({
        "record_id":  id_col,
        "price":      price_col,
        "event_date": date_col,
        "quantity":   rng.integers(1, 50, size=n).tolist(),
    })

    schema = {
        "price":      {"dtype": "float",      "unit": "USD",        "description": "All prices must be USD float"},
        "event_date": {"format": "YYYY-MM-DD", "description": "ISO 8601"},
        "record_id":  {"format": "8-char alphanumeric uppercase", "regex": "^[A-Z0-9]{8}$"},
    }

    return df, schema, gbp_to_usd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASK_GENERATORS = {
    "easy":   lambda seed=42: generate_easy(seed),
    "medium": lambda seed=42: generate_medium(seed),
    "hard":   lambda seed=42: generate_hard(seed),
}


if __name__ == "__main__":
    df_e, s_e = generate_easy()
    print("=== EASY ===")
    print(df_e.to_string())
    print("Schema:", s_e)

    df_m, s_m = generate_medium()
    print("\n=== MEDIUM ===")
    print(df_m.head(10).to_string())
    print("Nulls in email:", df_m["email"].isna().sum())
    print("Duplicates:", df_m.duplicated().sum())

    df_h, s_h, rate = generate_hard()
    print("\n=== HARD ===")
    print(df_h.head(10).to_string())
    print(f"GBP→USD rate: {rate}")