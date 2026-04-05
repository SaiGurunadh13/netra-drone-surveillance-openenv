from __future__ import annotations

from typing import Any

from .models import GraderResponse, MissionMetrics, RewardModel, StepInfo
from .tasks import HIGH_RISK, MEDIUM_RISK, TaskDefinition, apply_risk_updates, task_high_risk_targets

MOVE_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
MOVE_COST = 1
SCAN_COST = 1

def _in_bounds(task: TaskDefinition, position: tuple[int, int]) -> bool:
    row, col = position
    return 0 <= row < task.height and 0 <= col < task.width

def _clone_grid(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in grid]

def _count_unique_scans(visited_map: list[list[int]]) -> int:
    return sum(value for row in visited_map for value in row)

def _count_visited_high_targets(task: TaskDefinition, visited_map: list[list[int]]) -> int:
    return sum(1 for row, col in task_high_risk_targets(task) if visited_map[row][col] == 1)

def mission_metrics(
    task: TaskDefinition,
    visited_map: list[list[int]],
    battery_units_remaining: int,
    step_count: int,
    drone_position: tuple[int, int],
) -> MissionMetrics:
    targets_total = len(task_high_risk_targets(task))
    targets_visited = _count_visited_high_targets(task, visited_map)
    high_risk_coverage = targets_visited / targets_total if targets_total else 1.0
    battery_remaining_score = battery_units_remaining / task.battery_budget
    unique_cells_scanned = _count_unique_scans(visited_map)
    returned_to_base = drone_position == task.base_position
    battery_used = max(1, task.battery_budget - battery_units_remaining)
    efficiency = min(1.0, (unique_cells_scanned + (1 if returned_to_base else 0)) / battery_used)
    return MissionMetrics(
        high_risk_targets_total=targets_total,
        high_risk_targets_visited=targets_visited,
        high_risk_coverage=round(high_risk_coverage, 4),
        efficiency=round(efficiency, 4),
        battery_remaining_score=round(battery_remaining_score, 4),
        unique_cells_scanned=unique_cells_scanned,
        returned_to_base=returned_to_base,
    )

def final_score(
    task: TaskDefinition,
    visited_map: list[list[int]],
    battery_units_remaining: int,
    step_count: int,
    drone_position: tuple[int, int],
) -> float:
    metrics = mission_metrics(task, visited_map, battery_units_remaining, step_count, drone_position)
    score = (
        0.6 * metrics.high_risk_coverage
        + 0.2 * metrics.efficiency
        + 0.2 * metrics.battery_remaining_score
    )
    return round(max(0.0, min(1.0, score)), 4)

def simulate_action(
    task: TaskDefinition,
    *,
    action_name: str,
    drone_position: tuple[int, int],
    battery_units_remaining: int,
    risk_map: list[list[int]],
    visited_map: list[list[int]],
    step_index: int,
    cumulative_reward: float,
) -> tuple[RewardModel, StepInfo, dict[str, Any]]:
    next_position = drone_position
    next_battery_units = battery_units_remaining
    next_risk_map = _clone_grid(risk_map)
    next_visited_map = _clone_grid(visited_map)
    raw_reward = 0.0
    penalties: dict[str, float] = {}
    feedback: list[str] = []
    done_reason: str | None = None
    risk_updates_applied: list[str] = []

    if action_name in MOVE_DELTAS:
        delta_row, delta_col = MOVE_DELTAS[action_name]
        candidate = (drone_position[0] + delta_row, drone_position[1] + delta_col)
        next_battery_units = max(0, battery_units_remaining - MOVE_COST)
        raw_reward -= 0.5
        if _in_bounds(task, candidate):
            next_position = candidate
            feedback.append(f"Moved to {next_position}.")
        else:
            penalties["invalid_boundary_move"] = -0.3
            raw_reward -= 0.3
            feedback.append("Attempted to leave the grid boundary.")
    elif action_name == "SCAN":
        next_battery_units = max(0, battery_units_remaining - SCAN_COST)
        raw_reward -= 0.5
        row, col = drone_position
        if next_visited_map[row][col] == 0:
            next_visited_map[row][col] = 1
            raw_reward += 0.2
            feedback.append(f"Scanned new cell {drone_position}.")
        else:
            penalties["repeat_scan"] = -0.1
            raw_reward -= 0.1
            feedback.append(f"Cell {drone_position} was already scanned.")
        current_risk = next_risk_map[row][col]
        if current_risk == HIGH_RISK:
            raw_reward += 2.0
            feedback.append("High-risk zone scanned: +2.0 surveillance reward.")
        elif current_risk == MEDIUM_RISK:
            raw_reward += 1.0
            feedback.append("Medium-risk zone scanned: +1.0 surveillance reward.")
    elif action_name == "RETURN_BASE":
        distance = abs(drone_position[0] - task.base_position[0]) + abs(drone_position[1] - task.base_position[1])
        required_cost = distance
        if required_cost <= battery_units_remaining:
            next_battery_units = battery_units_remaining - required_cost
            raw_reward -= 0.5 * required_cost
            next_position = task.base_position
            done_reason = "returned_to_base"
            feedback.append("Drone returned safely to base.")
        else:
            next_battery_units = 0
            raw_reward -= 0.5 * battery_units_remaining
            penalties["failed_return"] = -1.0
            raw_reward -= 1.0
            done_reason = "battery_depleted"
            feedback.append("Drone attempted to return but ran out of battery before reaching base.")
    else:
        penalties["unknown_action"] = -0.5
        raw_reward -= 0.5
        feedback.append(f"Unsupported action: {action_name}")

    risk_updates_applied = apply_risk_updates(task, next_risk_map, step_index)
    if risk_updates_applied:
        feedback.append("Risk map updated after this step.")

    if done_reason is None:
        if next_battery_units <= 0:
            done_reason = "battery_depleted"
        elif step_index >= task.max_steps:
            done_reason = "max_steps_reached"

    if done_reason is not None:
        remaining_high_risk = len(task_high_risk_targets(task)) - _count_visited_high_targets(task, next_visited_map)
        if remaining_high_risk > 0:
            penalties["unvisited_high_risk_at_end"] = -2.0
            raw_reward -= 2.0
            feedback.append("Mission ended with unvisited high-risk zones.")

    next_cumulative_reward = round(cumulative_reward + raw_reward, 4)
    metrics = mission_metrics(task, next_visited_map, next_battery_units, step_index, next_position)
    normalized_final_score = final_score(task, next_visited_map, next_battery_units, step_index, next_position) if done_reason else None
    if normalized_final_score is not None:
        feedback.append(f"Normalized mission score: {normalized_final_score:.4f}.")

    info = StepInfo(
        task_id=task.task_id,
        difficulty=task.difficulty,
        step_index=step_index,
        action=action_name,
        battery_before=round(battery_units_remaining / task.battery_budget, 4),
        battery_after=round(next_battery_units / task.battery_budget, 4),
        raw_reward=round(raw_reward, 4),
        cumulative_reward=next_cumulative_reward,
        penalties=penalties,
        risk_updates_applied=risk_updates_applied,
        mission_metrics=metrics,
        final_score=normalized_final_score,
        done_reason=done_reason,
        feedback=feedback,
    )
    reward = RewardModel(
        value=round(raw_reward, 4),
        components={
            "cumulative_reward": next_cumulative_reward,
            "final_score": normalized_final_score or 0.0,
        },
    )
    updates = {
        "drone_position": next_position,
        "battery_units_remaining": next_battery_units,
        "risk_map": next_risk_map,
        "visited_map": next_visited_map,
        "done": done_reason is not None,
        "final_score": normalized_final_score,
    }
    return reward, info, updates

def grade_action_plan(task: TaskDefinition, action_plan: list[str]) -> GraderResponse:
    drone_position = task.base_position
    battery_units_remaining = task.battery_budget
    risk_map = [list(row) for row in task.initial_risk_map]
    visited_map = [[0 for _ in range(task.width)] for _ in range(task.height)]
    cumulative_reward = 0.0
    final_info: StepInfo | None = None

    for step_index, action_name in enumerate(action_plan, start=1):
        reward, final_info, updates = simulate_action(
            task,
            action_name=action_name,
            drone_position=drone_position,
            battery_units_remaining=battery_units_remaining,
            risk_map=risk_map,
            visited_map=visited_map,
            step_index=step_index,
            cumulative_reward=cumulative_reward,
        )
        drone_position = updates["drone_position"]
        battery_units_remaining = updates["battery_units_remaining"]
        risk_map = updates["risk_map"]
        visited_map = updates["visited_map"]
        cumulative_reward = final_info.cumulative_reward
        if updates["done"]:
            break

    if final_info is None:
        empty_metrics = mission_metrics(task, visited_map, battery_units_remaining, 0, drone_position)
        final_info = StepInfo(
            task_id=task.task_id,
            difficulty=task.difficulty,
            step_index=1,
            action="SCAN",
            battery_before=1.0,
            battery_after=1.0,
            raw_reward=0.0,
            cumulative_reward=0.0,
            penalties={"empty_plan": -0.5},
            risk_updates_applied=[],
            mission_metrics=empty_metrics,
            final_score=0.0,
            done_reason="empty_plan",
            feedback=["No actions were provided for grading."],
        )

    score = final_info.final_score or final_score(task, visited_map, battery_units_remaining, max(1, len(action_plan)), drone_position)
    return GraderResponse(
        task_id=task.task_id,
        reward=RewardModel(value=score, components={"score": score}),
        raw_score=score,
        done=bool(final_info.done_reason),
        info=final_info,
    )
