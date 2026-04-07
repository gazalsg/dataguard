---
title: Dataguard
emoji: 😻
colorFrom: red
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: OpenEnv RL environment for automated data sanitization & com
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

DataGuard is a reinforcement learning environment where an agent acts as a **Data Integrity Officer**. Given a corrupted dataset and a schema specification, the agent must select a sequence of tool actions to clean and validate the data — maximising a deterministic compliance score.

---

## Motivation

Data cleaning is the #1 "boring" task in data science — consuming an estimated 80% of a data scientist's time. Yet no dedicated RL environment exists for training agents to handle it. DataGuard fills this gap with:

- **Deterministic graders** — no LLM judge, no subjectivity. Either the date is ISO 8601 or it isn't.
- **Rich partial reward signals** — every fix is rewarded incrementally.
- **Realistic complexity** — mixed currencies, inconsistent date formats, corrupted IDs.

---

## Environment Description

### Action Space

The agent selects from 7 structured tool actions:

| Action               | Parameters                               | Description                                     |
| -------------------- | ---------------------------------------- | ----------------------------------------------- |
| `fix_dtype`          | `column`, `target_type`                  | Cast column to int/float/string/date            |
| `standardize_format` | `column`, `format`                       | Apply format rule (e.g. YYYY-MM-DD, Title Case) |
| `drop_duplicates`    | `subset`                                 | Remove duplicate rows                           |
| `fill_nulls`         | `column`, `strategy`                     | Fill nulls: mean/median/mode/drop               |
| `convert_units`      | `column`, `from_unit`, `to_unit`, `rate` | Convert numeric units (e.g. GBP → USD)          |
| `drop_rows`          | `condition`                              | Drop rows matching a pandas query condition     |
| `validate_schema`    | —                                        | Score current state and end episode             |

### Observation Space

Each step the agent receives:

- Dataset summary (shape, per-column dtype/nulls/sample values)
- Schema requirements (target formats, dtypes, constraints)
- Available actions list
- Cumulative reward so far
- Feedback from last action
- Optional hint

### Reward Function

Rewards are computed by a **deterministic Python grader** — not an LLM.

```
validate_schema() → score ∈ [0.0, 1.0]
```

- Partial credit per column/issue class resolved
- **Row retention penalty**: up to -0.20 if agent drops >10% of originally valid rows
- Score is final and non-reversible once `validate_schema` is called

---

## Tasks

### Task 1 — Easy

- **Data**: 10 rows, 4 columns
- **Errors**: Names are UPPERCASE (should be Title Case), dates are MM/DD/YYYY (should be YYYY-MM-DD)
- **Max steps**: 6
- **Reward**: 0.5 per fix class

### Task 2 — Medium

- **Data**: 100 rows, 4 columns
- **Errors**: 10% duplicate rows, ~15% null emails, age stored as float64 (should be int)
- **Max steps**: 10
- **Reward**: 0.33 / 0.34 / 0.33 per issue class

### Task 3 — Hard

- **Data**: 500 rows, 4 columns
- **Errors**: Mixed GBP/USD currency strings, 3 inconsistent date formats, ~20% corrupted record IDs
- **Exchange rate**: Provided in observation — agent must read and use it
- **Max steps**: 18
- **Reward**: 0.34 / 0.33 / 0.33 per issue class, -0.20 retention penalty

---

## Baseline Scores

| Task   | Raw Score | Solved Score |
| ------ | --------- | ------------ |
| easy   | ~0.00     | 1.00         |
| medium | ~0.00     | ~0.99        |
| hard   | ~0.05     | ~0.95        |

---

## Setup & Usage

### Prerequisites

- Docker
- Python 3.11+
- `pip install openenv-core`

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server/server.py

# In another terminal — run inference
export HF_TOKEN=your_token_here
python inference.py
```

### Docker

```bash
# Build
docker build -t dataguard .

# Run
docker run -p 7860:7860 dataguard

# Test
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task": "easy"}'
```

### Validate Submission

```bash
openenv validate
./validate-submission.sh https://your-space.hf.space
```

---

## API Endpoints

| Method | Path      | Description           |
| ------ | --------- | --------------------- |
| `GET`  | `/`       | Environment info      |
| `GET`  | `/health` | Health check          |
| `POST` | `/reset`  | Start new episode     |
| `POST` | `/step`   | Execute action        |
| `POST` | `/state`  | Raw environment state |
| `GET`  | `/tasks`  | List all tasks        |

### Example: Full Episode

```python
import requests

BASE = "http://localhost:7860"

# Reset
r = requests.post(f"{BASE}/reset", json={"task": "easy"}).json()
obs = r["observation"]

# Fix name casing
r = requests.post(f"{BASE}/step", json={
    "task": "easy",
    "action": {"action": "standardize_format", "column": "name", "format": "Title Case"}
}).json()

# Fix date format
r = requests.post(f"{BASE}/step", json={
    "task": "easy",
    "action": {"action": "standardize_format", "column": "signup_date", "format": "YYYY-MM-DD"}
}).json()

# Validate — ends episode, returns score
r = requests.post(f"{BASE}/step", json={
    "task": "easy",
    "action": {"action": "validate_schema"}
}).json()

print(r["observation"]["reward_so_far"])  # → 1.0
```

---

## Project Structure

```
data-guard/
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── inference.py          ← mandatory hackathon script
├── README.md
├── server/
│   ├── server.py         ← FastAPI HTTP server
│   ├── env.py            ← DataGuardEnv (step/reset/state)
│   ├── grader.py         ← deterministic schema validator
│   ├── dataset_gen.py    ← synthetic dirty dataset generator
│   └── models.py         ← Pydantic Action/Observation/Reward
└── tests/
    └── test_dataguard.py ← pytest test suite
```

---

## Environment Variables

| Variable        | Default                            | Description         |
| --------------- | ---------------------------------- | ------------------- |
| `API_BASE_URL`  | `https://router.huggingface.co/v1` | LLM API endpoint    |
| `MODEL_NAME`    | `Qwen/Qwen2.5-72B-Instruct`        | Model identifier    |
| `HF_TOKEN`      | —                                  | HuggingFace API key |
| `DATAGUARD_URL` | `http://localhost:7860`            | Running server URL  |
| `PORT`          | `7860`                             | Server listen port  |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

# data-guard

OpenEnv RL environment for automated data sanitization &amp; compliance. Agents act as Data Integrity Officers to clean corrupted datasets against schema rules. Built for Meta × Scaler Hackathon.
