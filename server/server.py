"""
DataGuard — FastAPI Server
Exposes /reset, /step, /state endpoints per OpenEnv HTTP spec.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.task.env import DataGuardEnv, StepResult
from server.task.models import DataGuardAction

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DataGuard OpenEnv",
    description="Automated Data Sanitisation & Compliance Lab — OpenEnv Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per task (keyed by task name)
_envs: Dict[str, DataGuardEnv] = {}


def _get_or_create_env(task: str) -> DataGuardEnv:
    if task not in _envs:
        _envs[task] = DataGuardEnv(task=task)
    return _envs[task]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    task:   str = "easy"
    action: DataGuardAction


class StateRequest(BaseModel):
    task: str = "easy"


def _serialize_result(result: StepResult) -> Dict[str, Any]:
    return {
        "observation": result.observation.model_dump(),
        "reward":      result.reward,
        "done":        result.done,
        "info":        result.info,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name":    "DataGuard",
        "tasks":   ["easy", "medium", "hard"],
        "version": "1.0.0",
        "status":  "ok",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    """Reset the environment and return the initial observation."""
    if req is None:
        req = ResetRequest()
    env = DataGuardEnv(task=req.task, seed=req.seed)
    _envs[req.task] = env
    result = env.reset()
    return _serialize_result(result)


@app.post("/step")
def step(req: StepRequest):
    """Execute one action and return the next observation + reward."""
    env = _envs.get(req.task)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active episode for task={req.task!r}. Call /reset first.",
        )
    try:
        result = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _serialize_result(result)


@app.post("/state")
def state(req: StateRequest = None):
    """Return raw environment state for debugging."""
    if req is None:
        req = StateRequest()
    env = _envs.get(req.task)
    if env is None:
        return {"error": f"No active episode for task={req.task!r}. Call /reset first."}
    return env.state()


@app.get("/tasks")
def list_tasks():
    """Enumerate available tasks with descriptions."""
    return {
        "tasks": [
            {
                "name":        "easy",
                "description": "10 rows — fix name casing and date format",
                "max_steps":   6,
                "reward_max":  1.0,
            },
            {
                "name":        "medium",
                "description": "100 rows — remove duplicates, fill nulls, fix age dtype",
                "max_steps":   10,
                "reward_max":  1.0,
            },
            {
                "name":        "hard",
                "description": "500 rows — mixed currency, inconsistent dates, corrupted IDs",
                "max_steps":   18,
                "reward_max":  1.0,
            },
        ]
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False,
    )
