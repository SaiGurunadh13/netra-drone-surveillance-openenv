from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from env.environment import NetraDroneSurveillanceEnvironment
from env.models import Action, ActionType, BaselineResponse, BaselineTaskResult, Observation
from env.tasks import HIGH_RISK, MEDIUM_RISK, TASKS, TaskDefinition

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_MAX_COMPLETION_TOKENS = 128
DEFAULT_OPENAI_MAX_RETRIES = 6
DEFAULT_OPENAI_RETRY_BASE_DELAY_S = 0.75

def _load_env_from_dotenv() -> None:
   

    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def _selected_tasks(task_ids: list[str] | None) -> list[TaskDefinition]:
    tasks = list(TASKS)
    if task_ids is None:
        return tasks

    known = {task.task_id for task in tasks}
    unknown = [task_id for task_id in task_ids if task_id not in known]
    if unknown:
        raise ValueError(f"Unknown task ids: {', '.join(unknown)}")
    return [task for task in tasks if task.task_id in task_ids]

def _current_risk(observation: Observation) -> int:
    row, col = observation.drone_position
    return observation.risk_map[row][col]

def _is_current_cell_scanned(observation: Observation) -> bool:
    row, col = observation.drone_position
    return observation.visited_map[row][col] == 1

def _unvisited_cells(observation: Observation, risk_level: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for row_index, row in enumerate(observation.risk_map):
        for col_index, risk in enumerate(row):
            if risk == risk_level and observation.visited_map[row_index][col_index] == 0:
                cells.append((row_index, col_index))
    return cells

def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])

def _move_toward(current: tuple[int, int], target: tuple[int, int]) -> ActionType:
    if current[0] < target[0]:
        return ActionType.DOWN
    if current[0] > target[0]:
        return ActionType.UP
    if current[1] < target[1]:
        return ActionType.RIGHT
    return ActionType.LEFT

def choose_action_heuristic(observation: Observation) -> ActionType:
    current = observation.drone_position
    base = observation.base_position
    distance_to_base = _manhattan(current, base)
    current_risk = _current_risk(observation)
    current_scanned = _is_current_cell_scanned(observation)

    high_targets = _unvisited_cells(observation, HIGH_RISK)
    medium_targets = _unvisited_cells(observation, MEDIUM_RISK)

    if current_risk == HIGH_RISK and not current_scanned:
        return ActionType.SCAN

    if observation.battery_units_remaining <= distance_to_base + 1:
        return ActionType.RETURN_BASE

    reachable_high_targets = [
        cell
        for cell in high_targets
        if _manhattan(current, cell) + _manhattan(cell, base) + 1 <= observation.battery_units_remaining
    ]
    if reachable_high_targets:
        target = min(reachable_high_targets, key=lambda cell: (_manhattan(current, cell), cell[0], cell[1]))
        if current == target:
            return ActionType.SCAN
        return _move_toward(current, target)

    if current_risk == MEDIUM_RISK and not current_scanned and not high_targets:
        return ActionType.SCAN

    reachable_medium_targets = [
        cell
        for cell in medium_targets
        if _manhattan(current, cell) + _manhattan(cell, base) + 1 <= observation.battery_units_remaining
    ]
    if reachable_medium_targets:
        target = min(reachable_medium_targets, key=lambda cell: (_manhattan(current, cell), cell[0], cell[1]))
        if current == target:
            return ActionType.SCAN
        return _move_toward(current, target)

    return ActionType.RETURN_BASE

def _resolve_baseline_mode(mode: str) -> str:
    if mode not in {"auto", "heuristic", "openai"}:
        raise ValueError(f"Unsupported baseline mode: {mode}")
    if mode == "auto":
        return "openai" if os.getenv("OPENAI_API_KEY") else "heuristic"
    return mode

def _build_openai_client() -> OpenAI:
    _load_env_from_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --mode openai.")

    client_kwargs: dict[str, str] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)

def _observation_payload(observation: Observation) -> dict[str, object]:
    return {
        "task_id": observation.task_id,
        "difficulty": observation.difficulty.value,
        "mission_brief": observation.mission_brief,
        "drone_position": list(observation.drone_position),
        "base_position": list(observation.base_position),
        "battery": observation.battery,
        "battery_units_remaining": observation.battery_units_remaining,
        "battery_capacity": observation.battery_capacity,
        "grid_size": list(observation.grid_size),
        "risk_map": observation.risk_map,
        "visited_map": observation.visited_map,
        "step_index": observation.step_index,
        "max_steps": observation.max_steps,
        "alerts": observation.alerts,
        "valid_actions": [action.value for action in observation.valid_actions],
    }

def _extract_action_name(text: str) -> str | None:
    for action_name in (action.value for action in ActionType):
        if re.search(rf"\b{re.escape(action_name)}\b", text.upper()):
            return action_name
    return None

def choose_action_openai(
    client: OpenAI,
    model: str,
    observation: Observation,
    action_history: list[ActionType],
    max_completion_tokens: int = DEFAULT_OPENAI_MAX_COMPLETION_TOKENS,
    max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
    retry_base_delay_s: float = DEFAULT_OPENAI_RETRY_BASE_DELAY_S,
) -> ActionType:
    def _request_action():
        return client.chat.completions.create(
            model=model,
            temperature=0,
            max_completion_tokens=max_completion_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You control a surveillance drone in a grid-based emergency monitoring task. "
                        "Choose exactly one next action from: UP, DOWN, LEFT, RIGHT, SCAN, RETURN_BASE. "
                        "Prioritize high-risk surveillance, preserve enough battery to return safely, and "
                        "respond with JSON only in the form {\"action\":\"ACTION_NAME\"}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "observation": _observation_payload(observation),
                            "previous_actions": [action.value for action in action_history],
                        }
                    ),
                },
            ],
        )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = _request_action()
            break
        except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            retry_after = None
            if getattr(exc, "response", None) is not None:
                retry_after = exc.response.headers.get("retry-after")
            if retry_after is not None:
                try:
                    delay_s = max(0.0, float(retry_after))
                except ValueError:
                    delay_s = retry_base_delay_s * (2**attempt)
            else:
                delay_s = retry_base_delay_s * (2**attempt)
            time.sleep(delay_s)
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Action request failed unexpectedly without an error.")

    content = response.choices[0].message.content or ""
    action_name = _extract_action_name(content)
    if action_name is None:
        raise ValueError(f"Could not parse action from OpenAI response: {content!r}")
    return ActionType(action_name)

def run_baseline(
    seed: int = 7,
    task_ids: list[str] | None = None,
    *,
    mode: str = "auto",
    model: str = DEFAULT_OPENAI_MODEL,
    max_completion_tokens: int = DEFAULT_OPENAI_MAX_COMPLETION_TOKENS,
    max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
    retry_base_delay_s: float = DEFAULT_OPENAI_RETRY_BASE_DELAY_S,
) -> BaselineResponse:
    tasks = _selected_tasks(task_ids)
    environment = NetraDroneSurveillanceEnvironment(seed=seed)
    results: list[BaselineTaskResult] = []
    baseline_mode = _resolve_baseline_mode(mode)
    client = _build_openai_client() if baseline_mode == "openai" else None

    for task in tasks:
        observation = environment.reset(task_id=task.task_id, seed=seed)
        while True:
            try:
                if baseline_mode == "openai":
                    assert client is not None
                    action = choose_action_openai(
                        client=client,
                        model=model,
                        observation=observation,
                        action_history=environment.state().action_history,
                        max_completion_tokens=max_completion_tokens,
                        max_retries=max_retries,
                        retry_base_delay_s=retry_base_delay_s,
                    )
                else:
                    action = choose_action_heuristic(observation)
            except Exception:
                if mode != "auto":
                    raise
                action = choose_action_heuristic(observation)

            step_result = environment.step(Action(command=action))
            observation = step_result.observation
            if step_result.done:
                break

        state = environment.state()
        grader_view = environment.grade(task.task_id, [action.value for action in state.action_history])
        results.append(
            BaselineTaskResult(
                task_id=task.task_id,
                difficulty=task.difficulty,
                title=task.title,
                score=grader_view.reward.value,
                cumulative_reward=state.cumulative_reward,
                action_plan=state.action_history,
                info=grader_view.info,
            )
        )

    total_score = round(sum(result.score for result in results), 4)
    average_score = round(total_score / len(results), 4) if results else 0.0
    return BaselineResponse(
        seed=seed,
        task_count=len(results),
        average_score=average_score,
        total_score=total_score,
        results=results,
    )

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Netra surveillance baseline on all tasks.")
    parser.add_argument("--seed", default=7, type=int, help="Deterministic task-order seed.")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Optional task id filter. Repeat to evaluate multiple tasks.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "heuristic", "openai"],
        default="auto",
        help="Use OpenAI when available, force the heuristic, or require OpenAI explicitly.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model name used when --mode openai or --mode auto selects OpenAI.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_OPENAI_MAX_COMPLETION_TOKENS,
        help="Maximum completion tokens for each OpenAI action decision call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_OPENAI_MAX_RETRIES,
        help="Retry attempts for rate-limit and transient API errors per action request.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=DEFAULT_OPENAI_RETRY_BASE_DELAY_S,
        help="Initial retry delay in seconds before exponential backoff.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON report.")
    return parser.parse_args(argv)

def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_baseline(
        seed=args.seed,
        task_ids=args.task_ids,
        mode=args.mode,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        retry_base_delay_s=args.retry_base_delay,
    )
    if args.pretty:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(json.dumps(result.model_dump()))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
