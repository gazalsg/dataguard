"""
DataGuard — Task: Hard
500 rows | Mixed currency + inconsistent dates + corrupted IDs
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import pandas as pd

from dataset_gen import generate_hard

TASK_NAME    = "hard"
MAX_STEPS    = 18
REWARD_MAX   = 1.0
DIFFICULTY   = "hard"

DESCRIPTION  = """
The agent receives a 500-row sales transaction table with three complex issues:

  1. PRICE COLUMN — mixed currency strings:
       Some values are "£12.50" (GBP), some are "$15.87" (USD string),
       some are bare floats already in USD.
       The agent must convert ALL to clean USD floats.
       The GBP→USD exchange rate is provided in schema_requirements._exchange_rate_hint.
       Correct approach:
         a) convert_units(column='price', rate=<gbp_rate>)   ← converts ALL values using rate
            (the tool strips £/$ symbols and multiplies by rate — for USD values already
             numeric this over-converts, so the agent should use fix_dtype first to isolate
             currency strings, OR use the provided rate only on GBP rows)
         b) fix_dtype(column='price', target_type='float')   ← ensure clean float dtype

  2. EVENT_DATE COLUMN — three mixed formats:
       YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY all present in same column.
       The agent must standardize to ISO 8601 (YYYY-MM-DD).
       Correct approach:
         standardize_format(column='event_date', format='YYYY-MM-DD')
         (pandas infer_format=True handles all three formats)

  3. RECORD_ID COLUMN — ~20% corrupted:
       Valid IDs are exactly 8 uppercase alphanumeric characters: ^[A-Z0-9]{8}$
       Corrupted IDs include: wrong length (3-5 chars), lowercase, or "CORRUPT_XXXX" strings.
       The agent must DROP rows with invalid IDs (not try to fix them).
       Correct approach:
         drop_rows(condition="record_id.str.len() != 8 or not record_id.str.match('[A-Z0-9]{8}')")
         ⚠ Dropping >10% of rows triggers a retention penalty (up to -0.20).
         Since ~20% of IDs are bad, the agent must drop them — the penalty is expected and
         unavoidable on this task. A perfect non-penalised score is not possible.

SCORING:
  price (USD float):     0.34
  event_date (ISO 8601): 0.33
  record_id (valid):     0.33
  retention penalty:     up to -0.20 (unavoidable on hard — ~20% rows dropped)
  
  Maximum achievable score ≈ 0.80 (after ~20% ID rows dropped)
  Frontier model target:    ≥ 0.65
"""

SCHEMA_REQUIREMENTS: Dict[str, Any] = {
    "price": {
        "dtype":       "float",
        "unit":        "USD",
        "description": "All prices must be clean USD floats (no £ or $ prefixes)",
    },
    "event_date": {
        "format":      "YYYY-MM-DD",
        "description": "ISO 8601 date string",
    },
    "record_id": {
        "format":      "8-char alphanumeric uppercase",
        "regex":       "^[A-Z0-9]{8}$",
        "description": "Exactly 8 uppercase letters or digits",
    },
}

REWARD_BREAKDOWN = {
    "price_usd_float":      0.34,
    "event_date_iso":       0.33,
    "record_id_valid":      0.33,
    "row_retention_penalty": "up to -0.20 if >10% rows dropped",
}


def load(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any], float]:
    """Return (dirty_df, schema_requirements, gbp_usd_rate) for this task."""
    df, schema, rate = generate_hard(seed)
    return df, schema, rate