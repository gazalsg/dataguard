"""
DataGuard — Mandatory Inference Script
=======================================
Runs an LLM agent against all 3 DataGuard tasks and emits structured
stdout logs in the required [START] / [STEP] / [END] format.

Environment variables:
  API_BASE_URL   LLM endpoint  (default: HuggingFace router)
  MODEL_NAME     Model to use  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       HuggingFace / API key
  DATAGUARD_URL  Running DataGuard server URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL    = os.getenv("API_BASE_URL",  "https://router.huggingface.co/v1")
MODEL_NAME      = os.getenv("MODEL_NAME",    "Qwen/Qwen2.5-72B-Instruct")
API_KEY         = os.getenv("HF_TOKEN")      or os.getenv("API_KEY", "")
DATAGUARD_URL   = os.getenv("DATAGUARD_URL", "http://localhost:7860").rstrip("/")
BENCHMARK       = "dataguard"
TEMPERATURE     = 0.2    # low temp for deterministic tool selection
MAX_TOKENS      = 512
SUCCESS_THRESHOLD = 0.7  # score >= 0.7 counts as success

TASKS = ["easy", "medium", "hard"]

MAX_STEPS = {
    "easy":   6,
    "medium": 10,
    "hard":   18,
}

# ---------------------------------------------------------------------------
# Logging helpers — MANDATORY FORMAT
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# DataGuard HTTP client
# ---------------------------------------------------------------------------

def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{DATAGUARD_URL}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_reset(task: str, seed: int = 42) -> Dict[str, Any]:
    return _post("/reset", {"task": task, "seed": seed})


def env_step(task: str, action: Dict[str, Any]) -> Dict[str, Any]:
    return _post("/step", {"task": task, "action": action})


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are DataGuard — an expert Data Integrity Officer agent.
You receive a corrupted dataset observation and must clean it step-by-step
using the available tools until the schema requirements are satisfied.

AVAILABLE ACTIONS (pick exactly one per turn):
  fix_dtype        — {"action": "fix_dtype",          "column": "<col>", "target_type": "int|float|string|date"}
  standardize_fmt  — {"action": "standardize_format", "column": "<col>", "format": "YYYY-MM-DD|Title Case|uppercase|lowercase"}
  drop_duplicates  — {"action": "drop_duplicates",    "subset": ["<col1>", ...]}
  fill_nulls       — {"action": "fill_nulls",         "column": "<col>", "strategy": "mean|median|mode|drop"}
  convert_units    — {"action": "convert_units",      "column": "<col>", "from_unit": "<X>", "to_unit": "<Y>", "rate": <float>}
  drop_rows        — {"action": "drop_rows",          "condition": "<pandas query string>"}
  validate_schema  — {"action": "validate_schema"} ← call this when you think the data is clean

RULES:
1. Respond with ONLY a valid JSON object for one action — no markdown, no explanation.
2. Inspect schema_requirements carefully before each action.
3. Call validate_schema when you believe all issues are resolved.
4. If a column has mixed currency strings (£ / $), use convert_units first, then fix_dtype to float.
5. Do not drop more than 10% of rows (triggers a retention penalty).
""").strip()


def build_user_prompt(obs: Dict[str, Any]) -> str:
    cols_info = "\n".join(
        f"  - {c['name']} ({c['dtype']}, nulls={c['null_count']}): sample={c['sample']}"
        for c in obs["columns"]
    )
    return textwrap.dedent(f"""
TASK: {obs['task_name']}  |  Step {obs['step']}  |  Reward so far: {obs['reward_so_far']:.2f}
Rows: {obs['total_rows']}

COLUMNS:
{cols_info}

SCHEMA REQUIREMENTS:
{json.dumps(obs['schema_requirements'], indent=2)}

LAST ACTION RESULT: {obs.get('last_action_result', 'None')}
HINT: {obs.get('hint') or 'None'}

Respond with ONE JSON action object.
""").strip()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def get_agent_action(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM and parse its JSON action."""
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip any accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] Agent error: {exc}", flush=True)
        # Fallback: call validate_schema to end episode gracefully
        return {"action": "validate_schema"}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task: str) -> None:
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = env_reset(task)
        obs    = result["observation"]

        for step in range(1, MAX_STEPS[task] + 1):
            if result.get("done"):
                break

            action_dict = get_agent_action(client, obs)
            action_str  = json.dumps(action_dict)

            error_msg = None
            try:
                result  = env_step(task, action_dict)
                obs     = result["observation"]
                reward  = float(result.get("reward", 0.0))
                done    = bool(result.get("done", False))
            except Exception as exc:
                reward    = 0.0
                done      = False
                error_msg = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        # Compute final score from cumulative reward in last observation
        score   = float(obs.get("reward_so_far", sum(rewards)))
        score   = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Verify server is up
    try:
        resp = requests.get(f"{DATAGUARD_URL}/health", timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[DEBUG] DataGuard server not reachable at {DATAGUARD_URL}: {exc}", flush=True)
        raise SystemExit(1)

    for task in TASKS:
        run_episode(client, task)
        time.sleep(1)  # brief pause between tasks


if __name__ == "__main__":
    main()