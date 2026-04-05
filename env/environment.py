from __future__ import annotations

import random
import uuid

from .grader import grade_action_plan, simulate_action
from .models import Action, EnvironmentState, RewardModel, StepResult, TransitionRecord
from .tasks import (
    TaskDefinition,
    build_task_catalog,
    clone_risk_map,
    empty_visited_map,
    ordered_task_ids,
    task_to_observation,
    task_to_preview,
)

class NetraDroneSurveillanceEnvironment:
    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._tasks = build_task_catalog()
        self._ordered_task_ids = ordered_task_ids()
        self._cursor = 0
        self._state = EnvironmentState(available_task_ids=self._ordered_task_ids)

    def reset(self, task_id: str | None = None, seed: int | None = None):
        if seed is not None:
            self._rng.seed(seed)

        selected_task_id = task_id or self._ordered_task_ids[self._cursor % len(self._ordered_task_ids)]
        if selected_task_id not in self._tasks:
            raise KeyError(f"Unknown task id: {selected_task_id}")

        if task_id is None:
            self._cursor += 1

        task = self._tasks[selected_task_id]
        risk_map = clone_risk_map(task)
        visited_map = empty_visited_map(task)
        observation = task_to_observation(
            task,
            drone_position=task.base_position,
            battery_units_remaining=task.battery_budget,
            risk_map=risk_map,
            visited_map=visited_map,
            step_index=0,
        )
        self._state = EnvironmentState(
            episode_id=str(uuid.uuid4()),
            current_task_id=task.task_id,
            current_difficulty=task.difficulty,
            step_count=0,
            max_steps=task.max_steps,
            done=False,
            drone_position=task.base_position,
            battery=1.0,
            battery_units_remaining=task.battery_budget,
            battery_capacity=task.battery_budget,
            base_position=task.base_position,
            grid_size=(task.width, task.height),
            risk_map=risk_map,
            visited_map=visited_map,
            cumulative_reward=0.0,
            final_score=None,
            current_observation=observation,
            last_action=None,
            last_reward=None,
            action_history=[],
            history=[],
            available_task_ids=self._ordered_task_ids,
        )
        return observation

    def step(self, action: Action) -> StepResult:
        if self._state.current_task_id is None or self._state.current_observation is None:
            raise RuntimeError("Environment has not been reset.")
        if self._state.done:
            raise RuntimeError("Episode already completed. Call reset() before stepping again.")

        task = self._tasks[self._state.current_task_id]
        reward_model, info, updates = simulate_action(
            task,
            action_name=action.command,
            drone_position=self._state.drone_position,
            battery_units_remaining=self._state.battery_units_remaining,
            risk_map=self._state.risk_map,
            visited_map=self._state.visited_map,
            step_index=self._state.step_count + 1,
            cumulative_reward=self._state.cumulative_reward,
        )
        next_observation = task_to_observation(
            task,
            drone_position=updates["drone_position"],
            battery_units_remaining=updates["battery_units_remaining"],
            risk_map=updates["risk_map"],
            visited_map=updates["visited_map"],
            step_index=self._state.step_count + 1,
        )
        transition = TransitionRecord(
            observation=next_observation,
            action=action,
            reward=reward_model.value,
            done=updates["done"],
            info=info,
        )

        self._state.step_count += 1
        self._state.done = updates["done"]
        self._state.drone_position = updates["drone_position"]
        self._state.battery_units_remaining = updates["battery_units_remaining"]
        self._state.battery = round(updates["battery_units_remaining"] / task.battery_budget, 4)
        self._state.risk_map = updates["risk_map"]
        self._state.visited_map = updates["visited_map"]
        self._state.cumulative_reward = info.cumulative_reward
        self._state.final_score = updates["final_score"]
        self._state.current_observation = next_observation
        self._state.last_action = action
        self._state.last_reward = RewardModel(value=reward_model.value, components=reward_model.components)
        self._state.action_history.append(action.command)
        self._state.history.append(transition)

        return StepResult(
            observation=next_observation,
            reward=reward_model.value,
            done=updates["done"],
            info=info.model_dump(),
        )

    def state(self) -> EnvironmentState:
        return self._state.model_copy(deep=True)

    def tasks(self) -> list:
        return [task_to_preview(self._tasks[task_id]) for task_id in self._ordered_task_ids]

    def grade(self, task_id: str, action_plan: list[str]):
        task = self.get_task(task_id)
        return grade_action_plan(task, action_plan)

    def get_task(self, task_id: str) -> TaskDefinition:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task id: {task_id}")
        return self._tasks[task_id]
