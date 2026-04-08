---
title: DataGuard
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
  - data-engineering
  - agent-environment
---

# DataGuard 🛡️

### Teaching AI agents to do the work nobody wants to do — but everyone needs done.

Data cleaning consumes an estimated **80% of a data scientist's time**. It's unglamorous, repetitive, and surprisingly difficult to automate well. DataGuard is a reinforcement learning environment built around this problem — giving agents a structured workspace to learn how to take a messy, real-world dataset and make it trustworthy.

The agent plays the role of a **Data Integrity Officer**. It receives a corrupted dataset alongside a schema specification, and must select a sequence of cleaning operations — fixing formats, resolving type mismatches, removing duplicates, handling nulls, and converting inconsistent units — until the data meets the required standard.

What makes this interesting for RL is the reward structure: every fix is measurable, every mistake is detectable, and partial progress is always acknowledged. There's no LLM judge involved — just a deterministic Python grader that either accepts the data or doesn't.

---

## Why this environment exists

Most RL environments for language agents are either games or narrow API tasks. There are very few that model the kind of **messy, stateful, tool-use workflows** that show up in real data engineering work.

DataGuard fills that gap. The tasks it presents happen every day in companies around the world — a CSV exported from a legacy system with inconsistent date formats, a database dump where currency values were stored as strings, a user table with 10% duplicate rows that somehow slipped through. An agent that can reliably handle these scenarios has real economic value.

---

## How it works

The environment exposes a simple HTTP API following the OpenEnv spec. At each step, the agent receives an **observation** — a structured summary of the current dataset including column types, null counts, sample values, and the target schema. It then selects one **action** from a constrained toolkit of data operations. The episode ends when the agent calls `validate_schema()`, at which point the grader scores the result.

### The action toolkit

| Action               | What it does                                                  |
| -------------------- | ------------------------------------------------------------- |
| `fix_dtype`          | Cast a column to int, float, string, or date                  |
| `standardize_format` | Apply a format rule — ISO 8601 dates, Title Case names, etc.  |
| `drop_duplicates`    | Remove duplicate rows by one or more key columns              |
| `fill_nulls`         | Fill missing values using mean, median, mode, or row deletion |
| `convert_units`      | Convert numeric values between units using a provided rate    |
| `drop_rows`          | Remove rows matching a pandas query condition                 |
| `validate_schema`    | Score the current state and close the episode                 |

Actions are intentionally constrained — no free-form code execution, no arbitrary transformations. This keeps the grader deterministic and prevents reward hacking.

---

## The three tasks

### Easy — First day on the job

**10 rows. Two problems. You've seen worse.**

A small customer table where someone exported names in ALL CAPS and dates in the American MM/DD/YYYY format instead of ISO 8601. A competent agent should solve this in 2–3 steps.

- Fix name casing: `ALICE JOHNSON` → `Alice Johnson`
- Fix date format: `06/11/2020` → `2020-06-11`
- Max steps: 6 | Reward: 0.5 per fix class

### Medium — Production data quality issue

**100 rows. Three compounding problems. Order matters.**

Closer to what you'd actually encounter in a real pipeline. Duplicate rows from a botched merge, ~15% of email addresses are NULL, and the age column was exported as float64 when it should be integer. The interesting challenge: the agent needs to decide the right order of operations.

- Remove duplicate rows
- Drop or fill null emails (must be valid format)
- Cast age from float to int, validate range 0–120
- Max steps: 10 | Reward: 0.33 / 0.34 / 0.33 per issue class

### Hard — The legacy system handoff

**500 rows. Three systemic issues. One of them is a trap.**

What happens when two systems that never talked to each other suddenly need to share data. Prices are stored as strings — some prefixed with `£`, some with `$`, some as bare floats. Dates appear in three different formats within the same column. About 20% of record IDs are corrupted — wrong length, lowercase, or structurally invalid.

The GBP→USD exchange rate is provided in the observation. The agent has to find it, use it, and clean up. Dropping bad ID rows is unavoidable — but dropping too many triggers a retention penalty.

- Convert mixed GBP/USD strings to clean USD floats
- Standardize three date formats to ISO 8601
- Drop rows with invalid 8-char alphanumeric IDs
- Max steps: 18 | Up to -0.20 retention penalty
- **Maximum achievable score: ~0.80** (the ID drops are unavoidable)

---

## Reward design

Rewards are computed entirely by a deterministic Python grader — no LLM, no heuristics, no subjectivity.

A few deliberate design decisions worth noting:

**Partial credit is always available.** An agent that fixes two out of three issues in the medium task scores ~0.67, not 0. This gives the RL training signal something to work with even in early episodes.

**Row retention is penalised.** An agent that "solves" the hard task by deleting everything technically passes the schema check. The retention penalty discourages this. A good agent learns to surgically remove only the rows that are actually bad.

**Rewards only arrive at `validate_schema()`**. No signal during intermediate steps — only the final graded result counts. This encourages the agent to plan ahead rather than greedy-step through the problem.

---

## Baseline scores

Achieved by `Qwen/Qwen2.5-72B-Instruct` via the HuggingFace router:

| Task   | Score | Notes                                         |
| ------ | ----- | --------------------------------------------- |
| easy   | 1.000 | Solved in 3 effective steps                   |
| medium | 0.977 | Missed duplicate removal, still strong        |
| hard   | 0.955 | Correctly used exchange rate from observation |

---

## Getting started

### Run locally

```bash
pip install -r requirements.txt
python app.py

# In a separate terminal
export HF_TOKEN=your_token_here
export DATAGUARD_URL=http://localhost:7860
python inference.py
```

### Run with Docker

```bash
docker build -t dataguard .
docker run -p 7860:7860 dataguard
```

### Validate before submitting

```bash
python validate_local.py --url http://localhost:7860
```

### Quick API test

```bash
curl https://gkk1-dataguard.hf.space/

curl -X POST https://gkk1-dataguard.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'
```

---

## API reference

| Method | Endpoint  | Description                                                   |
| ------ | --------- | ------------------------------------------------------------- | ------ | ------------------- |
| `GET`  | `/`       | Environment info and status                                   |
| `GET`  | `/health` | Health check                                                  |
| `POST` | `/reset`  | Start a new episode — body: `{"task": "easy                   | medium | hard", "seed": 42}` |
| `POST` | `/step`   | Execute one action — body: `{"task": "...", "action": {...}}` |
| `POST` | `/state`  | Raw environment state (useful for debugging)                  |
| `GET`  | `/tasks`  | List all tasks with metadata                                  |

---

## Environment variables

| Variable        | Default                            | Description            |
| --------------- | ---------------------------------- | ---------------------- |
| `HF_TOKEN`      | —                                  | Your HuggingFace token |
| `API_BASE_URL`  | `https://router.huggingface.co/v1` | LLM API endpoint       |
| `MODEL_NAME`    | `Qwen/Qwen2.5-72B-Instruct`        | Model identifier       |
| `DATAGUARD_URL` | `http://localhost:7860`            | DataGuard server URL   |
| `PORT`          | `7860`                             | Server listen port     |

---

## Project layout

```
dataguard/
├── app.py                  # Entry point (HF Spaces + local)
├── inference.py            # Baseline agent script
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── validate_local.py       # Pre-submission checklist
├── server/
│   ├── server.py           # FastAPI — /reset /step /state /tasks
│   ├── env.py              # Episode logic
│   ├── grader.py           # Deterministic schema validator
│   ├── dataset_gen.py      # Synthetic dataset generator
│   ├── models.py           # Pydantic models
│   └── tasks/
│       ├── easy.py
│       ├── medium.py
│       └── hard.py
└── tests/
    └── test_dataguard.py
```

---

_Built for the Meta PyTorch OpenEnv Hackathon. The environment, datasets, and graders are entirely original._
