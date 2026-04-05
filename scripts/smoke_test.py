"""Deterministic smoke test for the Netra Drone Surveillance environment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.environment import NetraDroneSurveillanceEnvironment
from env.models import Action
from env.tasks import TASKS


MIN_REFERENCE_SCORES = {
    "easy_static_risk_sweep": 0.65,
    "medium_limited_battery_dynamic_sweep": 0.60,
    "hard_fully_dynamic_response_grid": 0.25,
}


def main() -> int:
    env = NetraDroneSurveillanceEnvironment(seed=7)

    for task in TASKS:
        env.reset(task_id=task.task_id, seed=7)
        last_result = None
        for action in task.reference_action_plan:
            last_result = env.step(Action(command=action))
            if last_result.done:
                break

        assert last_result is not None
        assert last_result.done
        state = env.state()
        assert state.final_score is not None
        assert state.final_score >= MIN_REFERENCE_SCORES[task.task_id]
        print(
            f"{task.task_id}: score={state.final_score:.4f} reward={state.cumulative_reward:.2f} actions={len(state.action_history)}"
        )

    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
