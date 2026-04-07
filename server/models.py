"""
DataGuard Environment — Pydantic Models
Defines the typed Action, Observation, and Reward models for the OpenEnv spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action space enums — constrained to prevent reward hacking / free-form abuse
# ---------------------------------------------------------------------------

class FillStrategy(str, Enum):
    MEAN   = "mean"
    MEDIAN = "median"
    MODE   = "mode"
    DROP   = "drop"


class TargetDtype(str, Enum):
    INT    = "int"
    FLOAT  = "float"
    STRING = "string"
    DATE   = "date"    # ISO 8601: YYYY-MM-DD


class ActionType(str, Enum):
    FIX_DTYPE          = "fix_dtype"
    STANDARDIZE_FORMAT = "standardize_format"
    DROP_DUPLICATES    = "drop_duplicates"
    FILL_NULLS         = "fill_nulls"
    CONVERT_UNITS      = "convert_units"
    DROP_ROWS          = "drop_rows"
    VALIDATE_SCHEMA    = "validate_schema"


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class DataGuardAction(BaseModel):
    """
    One tool call the agent wants to execute.

    Examples
    --------
    fix_dtype:
        {"action": "fix_dtype", "column": "age", "target_type": "int"}

    standardize_format:
        {"action": "standardize_format", "column": "signup_date", "format": "YYYY-MM-DD"}

    drop_duplicates:
        {"action": "drop_duplicates", "subset": ["email"]}

    fill_nulls:
        {"action": "fill_nulls", "column": "salary", "strategy": "median"}

    convert_units:
        {"action": "convert_units", "column": "price", "from_unit": "GBP",
         "to_unit": "USD", "rate": 1.27}

    drop_rows:
        {"action": "drop_rows", "condition": "age < 0 or age > 120"}

    validate_schema:
        {"action": "validate_schema"}
    """

    action: ActionType

    # Shared optional params
    column: Optional[str]            = Field(None, description="Target column name")
    subset: Optional[List[str]]      = Field(None, description="Columns to check for duplicates")

    # fix_dtype
    target_type: Optional[TargetDtype] = Field(None, description="Cast column to this dtype")

    # standardize_format
    format: Optional[str]            = Field(None, description="Target format string (e.g. YYYY-MM-DD)")

    # fill_nulls
    strategy: Optional[FillStrategy] = Field(None, description="Null-fill strategy")

    # convert_units
    from_unit: Optional[str]         = Field(None, description="Source unit (e.g. GBP)")
    to_unit:   Optional[str]         = Field(None, description="Target unit (e.g. USD)")
    rate:      Optional[float]       = Field(None, description="Conversion multiplier")

    # drop_rows
    condition: Optional[str]         = Field(None, description="Pandas query string for rows to drop")


# ---------------------------------------------------------------------------
# Observation model — what the agent sees at every step
# ---------------------------------------------------------------------------

class ColumnSummary(BaseModel):
    name:        str
    dtype:       str
    null_count:  int
    sample:      List[Any] = Field(default_factory=list, description="Up to 5 sample values")


class DataGuardObservation(BaseModel):
    task_name:          str
    step:               int
    total_rows:         int
    total_cols:         int
    original_row_count: int               = Field(..., description="Row count at episode start — used for retention check")
    columns:            List[ColumnSummary]
    schema_requirements: Dict[str, Any]   = Field(..., description="The target schema the agent must satisfy")
    available_actions:  List[str]         = Field(..., description="Human-readable list of callable actions")
    reward_so_far:      float             = Field(0.0, description="Cumulative reward in this episode")
    last_action_result: Optional[str]     = Field(None, description="Feedback from the last action")
    hint:               Optional[str]     = Field(None, description="Optional hint injected by the environment")


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    dtype_fixes:       float = 0.0
    format_fixes:      float = 0.0
    duplicate_removal: float = 0.0
    null_handling:     float = 0.0
    unit_conversion:   float = 0.0
    schema_validation: float = 0.0
    row_retention_penalty: float = 0.0


class DataGuardReward(BaseModel):
    total:     float
    breakdown: RewardBreakdown
    message:   str