---
title: Netra Drone Surveillance Environment
emoji: 🚁
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - docker
  - fastapi
  - reinforcement-learning
  - surveillance
  - drone
---

# Netra Drone Surveillance Environment

Netra is a real-world OpenEnv environment where an AI agent controls a battery-constrained surveillance drone and decides where to fly next to maximize useful monitoring coverage over high-risk areas.

The environment is designed for disaster response, perimeter monitoring, and infrastructure watch scenarios where high-risk zones matter more than raw exploration and the mission only counts if the drone uses battery carefully enough to complete the sortie safely.

## Why This Environment Matters

This is not a toy game. It simulates a decision problem that human operators and autonomy systems actually face:

- which sector should the drone inspect next
- when should it spend time scanning instead of moving
- how should it react when risk shifts mid-mission
- when is it worth returning to base early to preserve the sortie

That makes it useful for training and evaluating planning agents, RL policies, and LLM-based controllers on realistic tradeoffs between coverage, urgency, and safety.

## OpenEnv Interface

The environment follows the standard OpenEnv simulation API.

- `reset(seed=None, task_id=None)` starts a mission and returns an initial observation
- `step(action)` executes one drone action and returns observation, reward, and done
- `state()` returns the full internal mission state

The official app entrypoint is [`server/app.py`](/d:/meta%20hackthon/server/app.py), and the OpenEnv manifest is [`openenv.yaml`](/d:/meta%20hackthon/openenv.yaml).

## Observation Space

Each observation is a typed Pydantic model with the core fields requested in the problem statement:

```python
{
  "drone_position": (x, y),
  "battery": float,
  "risk_map": [[...]],
  "visited_map": [[...]]
}
```

The full observation also includes:

- `task_id` and `difficulty`
- `mission_brief`
- `battery_units_remaining` and `battery_capacity`
- `base_position` and `grid_size`
- `step_index` and `max_steps`
- `dynamic_risk`
- `alerts`
- `valid_actions`

## Action Space

The drone action space is discrete and fixed:

- `UP`
- `DOWN`
- `LEFT`
- `RIGHT`
- `SCAN`
- `RETURN_BASE`

Behavior rules:

- movement consumes battery
- `SCAN` marks the current cell as visited and rewards higher-risk cells more strongly
- `RETURN_BASE` ends the episode safely
- episodes terminate on safe return, battery depletion, or max-step exhaustion

## Reward Shaping

The reward is dense and trajectory-aware so agents get useful signal before the end of the episode.

- `+2.0` for scanning a high-risk zone
- `+1.0` for scanning a medium-risk zone
- `+0.2` for scanning a previously unvisited area
- `-0.5` for battery usage on movement or scanning
- `-2.0` at mission end if high-risk zones remain unvisited

This encourages agents to cover the right places, not just move around randomly.

## Dynamic Risk Model

The risk map can change during a mission using deterministic task-defined updates. That keeps evaluation reproducible while still modeling real operational changes such as new alerts, shifting threat corridors, or fresh disaster-response priorities.

## Tasks

Three deterministic tasks are included, each with a programmatic grader that returns a score from `0.0` to `1.0`.

### Easy: `easy_static_risk_sweep`

- static risk map
- generous battery budget
- objective: cover all high-risk cells and return safely

### Medium: `medium_limited_battery_dynamic_sweep`

- limited battery
- mild deterministic risk changes
- objective: prioritize high-risk zones efficiently under tighter constraints

### Hard: `hard_fully_dynamic_response_grid`

- fully dynamic deterministic risk schedule
- very limited battery margin
- objective: maximize weighted high-risk coverage and still make a safe return

## Graders

Each task uses a deterministic grader based on the requested formula:

```text
score =
  0.6 * high_risk_coverage
+ 0.2 * efficiency
+ 0.2 * battery_remaining
```

Where:

- `high_risk_coverage` measures how much high-priority surveillance was completed
- `efficiency` rewards useful coverage relative to battery spent
- `battery_remaining` rewards finishing with safer reserve margins

The grader implementation lives in [`env/grader.py`](/d:/meta%20hackthon/env/grader.py).

## Project Structure

```text
.
+-- api/
+-- baseline/
+-- env/
+-- scripts/
+-- server/
+-- .dockerignore
+-- Dockerfile
+-- openenv.yaml
+-- pyproject.toml
+-- README.md
+-- requirements.txt
+-- uv.lock
```

Key files:

- [`env/environment.py`](/d:/meta%20hackthon/env/environment.py)
- [`env/tasks.py`](/d:/meta%20hackthon/env/tasks.py)
- [`env/models.py`](/d:/meta%20hackthon/env/models.py)
- [`server/netra_environment.py`](/d:/meta%20hackthon/server/netra_environment.py)
- [`server/app.py`](/d:/meta%20hackthon/server/app.py)
- [`baseline/run.py`](/d:/meta%20hackthon/baseline/run.py)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want the official CLI directly, the repo is also compatible with:

```bash
pip install "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git"
```

## Run Locally

Start the OpenEnv server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Or use the packaged entrypoint after installation:

```bash
python -m server.app
```

Example reset:

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 7, "task_id": "easy_static_risk_sweep"}'
```

Example step:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"command": "RIGHT"}}'
```

Example state:

```bash
curl http://localhost:8000/state
```

## Validation

Validate the local environment structure:

```bash
python scripts/validate_config.py
openenv validate . --json --verbose
```

Validate a live server:

```bash
openenv validate --url http://127.0.0.1:8000
```

## Baseline Agent

The baseline in [`baseline/run.py`](/d:/meta%20hackthon/baseline/run.py) now supports two execution modes:

- `openai`: uses the OpenAI Python client and reads credentials from `OPENAI_API_KEY`
- `heuristic`: uses the deterministic local policy for no-cost reproducible smoke testing
- `auto`: uses OpenAI when `OPENAI_API_KEY` is present, otherwise falls back to the heuristic baseline

Set your API key in the shell before running the OpenAI baseline:

```bash
export OPENAI_API_KEY=your_key
python -m baseline.run --mode openai --model gpt-4.1-mini --seed 7 --pretty
```

Or create a local `.env` file at the repository root:

```bash
cp .env.example .env
# then set OPENAI_API_KEY in .env
python -m baseline.run --mode openai --model gpt-4.1-mini --seed 7 --pretty
```

For local deterministic evaluation without API usage:

```bash
python -m baseline.run --mode heuristic --seed 7 --pretty
```

Current deterministic heuristic baseline scores with seed `7`:

- `easy_static_risk_sweep`: `0.6821`
- `medium_limited_battery_dynamic_sweep`: `0.4472`
- `hard_fully_dynamic_response_grid`: `0.4031`
- average score: `0.5108`

## Submission Inference

The required submission entrypoint is [`inference.py`](/d:/meta%20hackthon/inference.py) at the repository root.

It uses the OpenAI Python client and reads these environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Example:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export HF_TOKEN=your_api_key
python inference.py --seed 7
```

## Smoke Test

Run the deterministic smoke test:

```bash
python scripts/smoke_test.py
```

Current smoke test snapshot:

- `easy_static_risk_sweep`: `0.6838`
- `medium_limited_battery_dynamic_sweep`: `0.6273`
- `hard_fully_dynamic_response_grid`: `0.2650`

## How To Test Full Simulation Quality

Deterministic smoke test using reference plans:

```bash
python scripts/smoke_test.py
```

Baseline policy run using the reproducible heuristic:

```bash
python -m baseline.run --mode heuristic --seed 7 --pretty
```

LLM-driven run:

Set your OpenAI environment variables, then run the baseline in OpenAI mode:

```bash
export OPENAI_API_KEY=your_key
python -m baseline.run --mode openai --model gpt-4.1-mini --seed 7 --pretty
```

Files that implement the simulation logic:

- [`env/environment.py`](/d:/meta%20hackthon/env/environment.py)
- [`env/grader.py`](/d:/meta%20hackthon/env/grader.py)
- [`env/tasks.py`](/d:/meta%20hackthon/env/tasks.py)
- [`server/app.py`](/d:/meta%20hackthon/server/app.py)

## Docker

Build the container:

```bash
docker build -t netra-drone-env .
```

Run it:

```bash
docker run --rm -p 8000:8000 netra-drone-env
```

## Hugging Face Spaces

This repository is ready for deployment as a Docker Space tagged with `openenv`.

- port `8000` is exposed
- `/health` and `/healthz` are available
- the service exposes the standard OpenEnv simulation endpoints
- the repo contains `openenv.yaml`, `Dockerfile`, `pyproject.toml`, and `uv.lock`
- live Space: `https://huggingface.co/spaces/SAIGURUNADH/netra-drone-surveillance-environment`
- live app: `https://saigurunadh-netra-drone-surveillance-environment.hf.space`
- `openenv validate --url https://saigurunadh-netra-drone-surveillance-environment.hf.space` passes

Deploy with the included helper:

```bash
export HF_TOKEN=your_token
python scripts/deploy_hf_space.py \
  --repo-id your-username/netra-drone-surveillance-environment \
  --restart \
  --wait
```

The deployment helper in [`scripts/deploy_hf_space.py`](/d:/meta%20hackthon/scripts/deploy_hf_space.py) creates the Space if needed, uploads the repo, optionally restarts the remote build, and can wait for the runtime to reach a terminal stage.

