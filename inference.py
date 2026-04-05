from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from baseline.run import (
    DEFAULT_OPENAI_MAX_COMPLETION_TOKENS,
    DEFAULT_OPENAI_MAX_RETRIES,
    DEFAULT_OPENAI_RETRY_BASE_DELAY_S,
    _load_env_from_dotenv,
    _selected_tasks,
    choose_action_openai,
)
from env.environment import NetraDroneSurveillanceEnvironment
from env.models import Action, BaselineResponse, BaselineTaskResult


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the submission inference entrypoint for Netra.")
    parser.add_argument("--seed", default=7, type=int, help="Deterministic task-order seed.")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Optional task id filter. Repeat to evaluate multiple tasks.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_OPENAI_MAX_COMPLETION_TOKENS,
        help="Maximum completion tokens for each LLM action call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_OPENAI_MAX_RETRIES,
        help="Retry attempts for transient LLM errors per action request.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=DEFAULT_OPENAI_RETRY_BASE_DELAY_S,
        help="Initial retry delay in seconds before exponential backoff.",
    )
    return parser.parse_args(argv)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _make_client() -> tuple[OpenAI, str, str]:
    _load_env_from_dotenv()
    api_base_url = _require_env("API_BASE_URL")
    model_name = _require_env("MODEL_NAME")
    hf_token = _require_env("HF_TOKEN")
    return OpenAI(api_key=hf_token, base_url=api_base_url), model_name, api_base_url


def _emit(tag: str, payload: dict[str, object]) -> None:
    print(f"{tag} {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}", flush=True)


def run_inference(
    seed: int = 7,
    task_ids: list[str] | None = None,
    *,
    max_completion_tokens: int = DEFAULT_OPENAI_MAX_COMPLETION_TOKENS,
    max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
    retry_base_delay_s: float = DEFAULT_OPENAI_RETRY_BASE_DELAY_S,
) -> BaselineResponse:
    client, model_name, api_base_url = _make_client()
    tasks = _selected_tasks(task_ids)
    environment = NetraDroneSurveillanceEnvironment(seed=seed)
    results: list[BaselineTaskResult] = []

    _emit(
        "[START]",
        {
            "seed": seed,
            "task_count": len(tasks),
            "task_ids": [task.task_id for task in tasks],
            "model_name": model_name,
            "api_base_url": api_base_url,
            "script": str(Path(__file__).name),
        },
    )

    for task in tasks:
        observation = environment.reset(task_id=task.task_id, seed=seed)
        while True:
            action = choose_action_openai(
                client=client,
                model=model_name,
                observation=observation,
                action_history=environment.state().action_history,
                max_completion_tokens=max_completion_tokens,
                max_retries=max_retries,
                retry_base_delay_s=retry_base_delay_s,
            )
            step_result = environment.step(Action(command=action))
            observation = step_result.observation
            info = step_result.info
            _emit(
                "[STEP]",
                {
                    "task_id": task.task_id,
                    "difficulty": task.difficulty.value,
                    "step_index": info["step_index"],
                    "action": action.value,
                    "reward": step_result.reward,
                    "done": step_result.done,
                    "battery": observation.battery,
                    "battery_units_remaining": observation.battery_units_remaining,
                    "drone_position": list(observation.drone_position),
                },
            )
            if step_result.done:
                break

        state = environment.state()
        grader_view = environment.grade(task.task_id, [action_taken.value for action_taken in state.action_history])
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
    response = BaselineResponse(
        seed=seed,
        task_count=len(results),
        average_score=average_score,
        total_score=total_score,
        results=results,
    )
    _emit(
        "[END]",
        {
            "seed": response.seed,
            "task_count": response.task_count,
            "average_score": response.average_score,
            "total_score": response.total_score,
            "results": [
                {
                    "task_id": result.task_id,
                    "difficulty": result.difficulty.value,
                    "score": result.score,
                    "cumulative_reward": result.cumulative_reward,
                }
                for result in response.results
            ],
        },
    )
    return response


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    run_inference(
        seed=args.seed,
        task_ids=args.task_ids,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        retry_base_delay_s=args.retry_base_delay,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
