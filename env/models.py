from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.types import (
        Action as OpenEnvAction,
        Observation as OpenEnvObservation,
        State as OpenEnvState,
    )
except Exception:  # pragma: no cover - fallback for local baseline-only runs
    class OpenEnvAction(BaseModel):
        model_config = ConfigDict(extra="allow")

    class OpenEnvObservation(BaseModel):
        model_config = ConfigDict(extra="allow")

        done: bool = False
        reward: float | None = None
        metadata: dict[str, Any] | None = None

    class OpenEnvState(BaseModel):
        model_config = ConfigDict(extra="allow")

        episode_id: str | None = None
        step_count: int = 0

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class ActionType(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    SCAN = "SCAN"
    RETURN_BASE = "RETURN_BASE"

class Observation(OpenEnvObservation):
    task_id: str
    difficulty: Difficulty
    title: str
    mission_brief: str
    drone_position: tuple[int, int]
    battery: float = Field(ge=0.0, le=1.0)
    battery_units_remaining: int = Field(ge=0)
    battery_capacity: int = Field(ge=1)
    base_position: tuple[int, int]
    grid_size: tuple[int, int]
    risk_map: list[list[int]] = Field(default_factory=list)
    visited_map: list[list[int]] = Field(default_factory=list)
    step_index: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    dynamic_risk: bool = False
    alerts: list[str] = Field(default_factory=list)
    valid_actions: list[ActionType] = Field(default_factory=list)

class Action(OpenEnvAction):
    command: ActionType

class RewardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    components: dict[str, float] = Field(default_factory=dict)

class TaskPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    title: str
    mission_brief: str
    grid_size: tuple[int, int]
    base_position: tuple[int, int]
    battery_budget: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    dynamic_risk: bool = False
    initial_risk_map: list[list[int]] = Field(default_factory=list)

class MissionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high_risk_targets_total: int = Field(ge=0)
    high_risk_targets_visited: int = Field(ge=0)
    high_risk_coverage: float = Field(ge=0.0, le=1.0)
    efficiency: float = Field(ge=0.0, le=1.0)
    battery_remaining_score: float = Field(ge=0.0, le=1.0)
    unique_cells_scanned: int = Field(ge=0)
    returned_to_base: bool = False

class StepInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    step_index: int = Field(ge=1)
    action: ActionType
    battery_before: float = Field(ge=0.0, le=1.0)
    battery_after: float = Field(ge=0.0, le=1.0)
    raw_reward: float
    cumulative_reward: float
    penalties: dict[str, float] = Field(default_factory=dict)
    risk_updates_applied: list[str] = Field(default_factory=list)
    mission_metrics: MissionMetrics
    final_score: float | None = Field(default=None, ge=0.0, le=1.0)
    done_reason: str | None = None
    feedback: list[str] = Field(default_factory=list)

class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)

class TransitionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    action: Action
    reward: float
    done: bool
    info: StepInfo

class EnvironmentState(OpenEnvState):
    model_config = ConfigDict(extra="forbid")

    current_task_id: str | None = None
    current_difficulty: Difficulty | None = None
    max_steps: int = Field(default=1, ge=1)
    done: bool = False
    drone_position: tuple[int, int] = (0, 0)
    battery: float = Field(default=0.0, ge=0.0, le=1.0)
    battery_units_remaining: int = Field(default=0, ge=0)
    battery_capacity: int = Field(default=1, ge=1)
    base_position: tuple[int, int] = (0, 0)
    grid_size: tuple[int, int] = (1, 1)
    risk_map: list[list[int]] = Field(default_factory=list)
    visited_map: list[list[int]] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    final_score: float | None = Field(default=None, ge=0.0, le=1.0)
    current_observation: Observation | None = None
    last_action: Action | None = None
    last_reward: RewardModel | None = None
    action_history: list[ActionType] = Field(default_factory=list)
    history: list[TransitionRecord] = Field(default_factory=list)
    available_task_ids: list[str] = Field(default_factory=list)

class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    seed: int | None = None

class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action

class GraderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    action_plan: list[ActionType] = Field(default_factory=list)

class GraderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    reward: RewardModel
    raw_score: float
    done: bool
    info: StepInfo

class BaselineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_ids: list[str] | None = None
    seed: int = 7

class BaselineTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    title: str
    score: float = Field(ge=0.0, le=1.0)
    cumulative_reward: float
    action_plan: list[ActionType] = Field(default_factory=list)
    info: StepInfo

class BaselineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    task_count: int = Field(ge=0)
    average_score: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(ge=0.0)
    results: list[BaselineTaskResult] = Field(default_factory=list)
