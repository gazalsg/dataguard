#!/usr/bin/env python3
"""
DataGuard — Local Pre-Submission Validator
==========================================
Runs all checks that the hackathon automated validator will run, locally.

Usage:
    python validate_local.py
    python validate_local.py --url http://localhost:7860

Checks:
  1. openenv.yaml exists and has required fields
  2. inference.py exists at root
  3. Dockerfile exists
  4. requirements.txt exists
  5. All 3 tasks can reset() and step() without error
  6. Graders return scores in [0.0, 1.0] for all tasks
  7. Server /health endpoint responds (if --url provided)
  8. Server /reset endpoint responds with 200 (if --url provided)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from typing import List, Tuple

import requests
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(ROOT, "server")
sys.path.insert(0, SERVER_DIR)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results: List[Tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    icon = PASS if ok else FAIL
    msg  = f"{icon}  {label}"
    if detail:
        msg += f"\n     {detail}"
    print(msg)
    results.append((ok, label))


# ---------------------------------------------------------------------------
# Check 1 — File existence
# ---------------------------------------------------------------------------

def check_files():
    print("\n── File Structure ──────────────────────────────")
    required = [
        ("openenv.yaml",   "openenv.yaml"),
        ("inference.py",   "inference.py (mandatory — must be at root)"),
        ("Dockerfile",     "Dockerfile"),
        ("requirements.txt", "requirements.txt"),
        ("server/env.py",  "server/env.py"),
        ("server/server.py", "server/server.py"),
        ("server/grader.py", "server/grader.py"),
        ("server/models.py", "server/models.py"),
        ("server/dataset_gen.py", "server/dataset_gen.py"),
    ]
    for rel_path, label in required:
        full = os.path.join(ROOT, rel_path)
        check(os.path.exists(full), label)


# ---------------------------------------------------------------------------
# Check 2 — openenv.yaml fields
# ---------------------------------------------------------------------------

def check_openenv_yaml():
    print("\n── openenv.yaml Validation ─────────────────────")
    path = os.path.join(ROOT, "openenv.yaml")
    if not os.path.exists(path):
        check(False, "openenv.yaml — skipped (file missing)")
        return

    with open(path) as f:
        cfg = yaml.safe_load(f)

    required_fields = ["name", "version", "description", "tasks"]
    for field in required_fields:
        check(field in cfg, f"openenv.yaml has '{field}' field")

    if "tasks" in cfg:
        tasks = cfg["tasks"]
        check(len(tasks) >= 3, f"openenv.yaml has ≥3 tasks (found {len(tasks)})")
        for t in tasks:
            for key in ["name", "difficulty", "max_steps", "reward_range"]:
                check(key in t, f"task '{t.get('name','?')}' has '{key}' field")


# ---------------------------------------------------------------------------
# Check 3 — Import server modules
# ---------------------------------------------------------------------------

def check_imports():
    print("\n── Module Imports ──────────────────────────────")
    modules = ["models", "dataset_gen", "grader", "env", "server"]
    for mod in modules:
        try:
            importlib.import_module(mod)
            check(True, f"import {mod}")
        except Exception as exc:
            check(False, f"import {mod}", str(exc))


# ---------------------------------------------------------------------------
# Check 4 — Environment lifecycle
# ---------------------------------------------------------------------------

def check_env_lifecycle():
    print("\n── Environment Lifecycle ───────────────────────")
    try:
        from env import DataGuardEnv
        from models import ActionType, DataGuardAction

        for task in ["easy", "medium", "hard"]:
            try:
                env    = DataGuardEnv(task)
                result = env.reset()
                obs    = result.observation

                # Step with a no-op validate_schema
                step_result = env.step(DataGuardAction(action=ActionType.VALIDATE_SCHEMA))
                score = step_result.observation.reward_so_far

                check(
                    True,
                    f"task={task} reset()+step() lifecycle",
                    f"score={score:.3f} done={step_result.done}",
                )
            except Exception as exc:
                check(False, f"task={task} lifecycle", traceback.format_exc(limit=3))
    except ImportError as exc:
        check(False, "DataGuardEnv import", str(exc))


# ---------------------------------------------------------------------------
# Check 5 — Grader scores are in [0, 1]
# ---------------------------------------------------------------------------

def check_grader_scores():
    print("\n── Grader Score Range ──────────────────────────")
    try:
        from dataset_gen import generate_easy, generate_medium, generate_hard
        from grader import grade

        for task, gen in [
            ("easy",   lambda: generate_easy()),
            ("medium", lambda: generate_medium()),
            ("hard",   lambda: (generate_hard()[0], generate_hard()[1])),
        ]:
            try:
                if task == "hard":
                    df, _, _ = generate_hard()
                elif task == "medium":
                    df, _ = generate_medium()
                else:
                    df, _ = generate_easy()

                result = grade(task, df, len(df))
                in_range = 0.0 <= result.total <= 1.0
                check(in_range, f"grader({task}) score in [0,1]", f"score={result.total:.4f}")
            except Exception as exc:
                check(False, f"grader({task})", str(exc))
    except ImportError as exc:
        check(False, "grader import", str(exc))


# ---------------------------------------------------------------------------
# Check 6 — HTTP server (optional, if --url given)
# ---------------------------------------------------------------------------

def check_server(url: str):
    print(f"\n── HTTP Server ({url}) ──────────────────")
    url = url.rstrip("/")

    # Health
    try:
        r = requests.get(f"{url}/health", timeout=10)
        check(r.status_code == 200, "GET /health → 200", f"status={r.status_code}")
    except Exception as exc:
        check(False, "GET /health", str(exc))
        return

    # Reset all tasks
    for task in ["easy", "medium", "hard"]:
        try:
            r = requests.post(
                f"{url}/reset",
                json={"task": task, "seed": 42},
                timeout=15,
            )
            ok = r.status_code == 200
            body = r.json() if ok else {}
            has_obs = "observation" in body
            check(ok and has_obs, f"POST /reset task={task} → 200 + observation")
        except Exception as exc:
            check(False, f"POST /reset task={task}", str(exc))

    # Step
    try:
        requests.post(f"{url}/reset", json={"task": "easy"}, timeout=10)
        r = requests.post(
            f"{url}/step",
            json={"task": "easy", "action": {"action": "validate_schema"}},
            timeout=15,
        )
        ok = r.status_code == 200
        body = r.json() if ok else {}
        check(ok and "reward" in body, "POST /step → 200 + reward field")
    except Exception as exc:
        check(False, "POST /step", str(exc))

    # Tasks listing
    try:
        r = requests.get(f"{url}/tasks", timeout=10)
        ok = r.status_code == 200
        tasks = r.json().get("tasks", []) if ok else []
        check(ok and len(tasks) == 3, f"GET /tasks → 3 tasks (found {len(tasks)})")
    except Exception as exc:
        check(False, "GET /tasks", str(exc))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    print("\n── Summary ─────────────────────────────────────")
    passed = sum(1 for ok, _ in results if ok)
    total  = len(results)
    failed = [(label) for ok, label in results if not ok]

    print(f"  {passed}/{total} checks passed")
    if failed:
        print("\n  Failed checks:")
        for label in failed:
            print(f"    {FAIL}  {label}")
    else:
        print(f"\n  {PASS}  All checks passed — ready to submit!")

    return len(failed) == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataGuard pre-submission validator")
    parser.add_argument("--url", default=None, help="Running server URL (e.g. http://localhost:7860)")
    args = parser.parse_args()

    print("=" * 52)
    print("  DataGuard — Pre-Submission Validator")
    print("=" * 52)

    check_files()
    check_openenv_yaml()
    check_imports()
    check_env_lifecycle()
    check_grader_scores()

    if args.url:
        check_server(args.url)
    else:
        print(f"\n{WARN}  Server checks skipped (pass --url http://localhost:7860 to include)")

    ok = print_summary()
    sys.exit(0 if ok else 1)
