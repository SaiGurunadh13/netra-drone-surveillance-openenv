from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .models import ActionType, Difficulty, Observation, TaskPreview

LOW_RISK = 0
MEDIUM_RISK = 1
HIGH_RISK = 2

@dataclass(frozen=True)
class RiskUpdate:
    step: int
    position: tuple[int, int]
    new_risk: int

@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    difficulty: Difficulty
    title: str
    mission_brief: str
    width: int
    height: int
    base_position: tuple[int, int]
    battery_budget: int
    max_steps: int
    initial_risk_map: tuple[tuple[int, ...], ...]
    risk_updates: tuple[RiskUpdate, ...] = ()
    reference_action_plan: tuple[ActionType, ...] = ()
    alerts: tuple[str, ...] = ()

    @property
    def dynamic_risk(self) -> bool:
        return bool(self.risk_updates)

def _grid(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(value for value in row) for row in rows)

TASKS: tuple[TaskDefinition, ...] = (
    TaskDefinition(
        task_id="easy_static_risk_sweep",
        difficulty=Difficulty.EASY,
        title="Static perimeter sweep over known high-risk blocks",
        mission_brief=(
            "The drone must inspect a static map of suspected activity around a disaster-relief corridor. "
            "Battery is generous, so the agent should cover every high-risk zone and return safely."
        ),
        width=6,
        height=6,
        base_position=(0, 0),
        battery_budget=24,
        max_steps=30,
        initial_risk_map=_grid(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 2, 0, 1, 0],
                [0, 0, 0, 1, 2, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 2, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        reference_action_plan=(
            ActionType.RIGHT,
            ActionType.RIGHT,
            ActionType.DOWN,
            ActionType.SCAN,
            ActionType.RIGHT,
            ActionType.RIGHT,
            ActionType.DOWN,
            ActionType.SCAN,
            ActionType.LEFT,
            ActionType.LEFT,
            ActionType.LEFT,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.SCAN,
            ActionType.RETURN_BASE,
        ),
        alerts=(
            "All high-risk cells are known at launch time.",
            "Mission success requires full high-risk coverage and a safe return.",
        ),
    ),
    TaskDefinition(
        task_id="medium_limited_battery_dynamic_sweep",
        difficulty=Difficulty.MEDIUM,
        title="Battery-limited infrastructure patrol with mild risk changes",
        mission_brief=(
            "A drone is monitoring critical infrastructure after reports of suspicious movement. "
            "Battery is tighter, and a new high-risk area may emerge during the patrol, so the route must stay efficient."
        ),
        width=7,
        height=7,
        base_position=(0, 0),
        battery_budget=22,
        max_steps=28,
        initial_risk_map=_grid(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 2, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        risk_updates=(
            RiskUpdate(step=4, position=(6, 5), new_risk=2),
            RiskUpdate(step=8, position=(2, 6), new_risk=1),
        ),
        reference_action_plan=(
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.RIGHT,
            ActionType.RIGHT,
            ActionType.SCAN,
            ActionType.UP,
            ActionType.UP,
            ActionType.UP,
            ActionType.RIGHT,
            ActionType.RIGHT,
            ActionType.RIGHT,
            ActionType.SCAN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.SCAN,
            ActionType.RETURN_BASE,
        ),
        alerts=(
            "A new high-risk sector may appear after the patrol begins.",
            "Battery is limited enough that inefficient movement will lower the final score quickly.",
        ),
    ),
    TaskDefinition(
        task_id="hard_fully_dynamic_response_grid",
        difficulty=Difficulty.HARD,
        title="Fully dynamic disaster-response surveillance grid",
        mission_brief=(
            "The drone is supporting emergency response over a dynamic surveillance grid. "
            "High-risk zones evolve during the mission, battery is tight, and the agent must maximize weighted coverage while still returning safely."
        ),
        width=8,
        height=8,
        base_position=(0, 0),
        battery_budget=24,
        max_steps=30,
        initial_risk_map=_grid(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        risk_updates=(
            RiskUpdate(step=3, position=(6, 6), new_risk=2),
            RiskUpdate(step=6, position=(3, 4), new_risk=2),
            RiskUpdate(step=9, position=(0, 5), new_risk=2),
            RiskUpdate(step=11, position=(4, 4), new_risk=0),
        ),
        reference_action_plan=(
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.RIGHT,
            ActionType.SCAN,
            ActionType.UP,
            ActionType.UP,
            ActionType.UP,
            ActionType.UP,
            ActionType.RIGHT,
            ActionType.RIGHT,
            ActionType.RIGHT,
            ActionType.RIGHT,
            ActionType.RIGHT,
            ActionType.SCAN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.DOWN,
            ActionType.RIGHT,
            ActionType.SCAN,
            ActionType.RETURN_BASE,
        ),
        alerts=(
            "Risk updates continue throughout the mission, so old routes can become stale.",
            "The mission is scored on weighted high-risk coverage, efficiency, and remaining battery.",
            "Safe return is strongly preferred on the hard task.",
        ),
    ),
)

def build_task_catalog() -> dict[str, TaskDefinition]:
    return {task.task_id: task for task in TASKS}

def ordered_task_ids() -> list[str]:
    return [task.task_id for task in TASKS]

def clone_risk_map(task: TaskDefinition) -> list[list[int]]:
    return [list(row) for row in task.initial_risk_map]

def empty_visited_map(task: TaskDefinition) -> list[list[int]]:
    return [[0 for _ in range(task.width)] for _ in range(task.height)]

def task_high_risk_targets(task: TaskDefinition) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    for row_index, row in enumerate(task.initial_risk_map):
        for col_index, risk in enumerate(row):
            if risk == HIGH_RISK:
                targets.add((row_index, col_index))
    for update in task.risk_updates:
        if update.new_risk == HIGH_RISK:
            targets.add(update.position)
    return targets

def apply_risk_updates(task: TaskDefinition, risk_map: list[list[int]], step_index: int) -> list[str]:
    applied: list[str] = []
    for update in task.risk_updates:
        if update.step == step_index:
            row, col = update.position
            risk_map[row][col] = update.new_risk
            applied.append(f"step {step_index}: ({row}, {col}) -> {update.new_risk}")
    return applied

def valid_actions() -> list[ActionType]:
    return [
        ActionType.UP,
        ActionType.DOWN,
        ActionType.LEFT,
        ActionType.RIGHT,
        ActionType.SCAN,
        ActionType.RETURN_BASE,
    ]

def task_to_observation(
    task: TaskDefinition,
    *,
    drone_position: tuple[int, int],
    battery_units_remaining: int,
    risk_map: list[list[int]],
    visited_map: list[list[int]],
    step_index: int,
) -> Observation:
    return Observation(
        task_id=task.task_id,
        difficulty=task.difficulty,
        title=task.title,
        mission_brief=task.mission_brief,
        drone_position=drone_position,
        battery=round(battery_units_remaining / task.battery_budget, 4),
        battery_units_remaining=battery_units_remaining,
        battery_capacity=task.battery_budget,
        base_position=task.base_position,
        grid_size=(task.width, task.height),
        risk_map=[list(row) for row in risk_map],
        visited_map=[list(row) for row in visited_map],
        step_index=step_index,
        max_steps=task.max_steps,
        dynamic_risk=task.dynamic_risk,
        alerts=list(task.alerts),
        valid_actions=valid_actions(),
    )

def task_to_preview(task: TaskDefinition) -> TaskPreview:
    return TaskPreview(
        task_id=task.task_id,
        difficulty=task.difficulty,
        title=task.title,
        mission_brief=task.mission_brief,
        grid_size=(task.width, task.height),
        base_position=task.base_position,
        battery_budget=task.battery_budget,
        max_steps=task.max_steps,
        dynamic_risk=task.dynamic_risk,
        initial_risk_map=clone_risk_map(task),
    )
