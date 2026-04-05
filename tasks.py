"""Compatibility wrapper for Netra task definitions and grading helpers."""

from env.grader import final_score, grade_action_plan, mission_metrics, simulate_action
from env.tasks import TASKS, TaskDefinition, build_task_catalog, ordered_task_ids

__all__ = [
    "TASKS",
    "TaskDefinition",
    "build_task_catalog",
    "ordered_task_ids",
    "simulate_action",
    "mission_metrics",
    "final_score",
    "grade_action_plan",
]
