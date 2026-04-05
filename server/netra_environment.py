from __future__ import annotations

from pathlib import Path

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from baseline.run import run_baseline
from env.environment import NetraDroneSurveillanceEnvironment
from env.models import Action, EnvironmentState, Observation

class NetraDroneSurveillanceServer(Environment[Action, Observation, EnvironmentState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._environment = NetraDroneSurveillanceEnvironment()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> Observation:
        observation = self._environment.reset(task_id=kwargs.get("task_id"), seed=seed)
        if episode_id is not None:
            self._environment._state.episode_id = episode_id
        return observation.model_copy(update={"done": False, "reward": None})

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs,
    ) -> Observation:
        del timeout_s, kwargs
        result = self._environment.step(action)
        return result.observation.model_copy(
            update={
                "reward": result.reward,
                "done": result.done,
                "metadata": result.info,
            }
        )

    @property
    def state(self) -> EnvironmentState:
        return self._environment.state()

    def get_metadata(self) -> EnvironmentMetadata:
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
        return EnvironmentMetadata(
            name="Netra Drone Surveillance Environment",
            description=(
                "A real-world surveillance planning environment where an agent routes "
                "a battery-constrained drone to monitor dynamic high-risk zones."
            ),
            version="5.0.0",
            readme_content=readme_content,
            author="OpenEnv Hackathon Submission",
        )

    def tasks(self):
        return self._environment.tasks()

    def grade(self, task_id: str, action_plan: list[str]):
        return self._environment.grade(task_id, action_plan)

    def baseline(self, seed: int = 7, task_ids: list[str] | None = None):
        return run_baseline(seed=seed, task_ids=task_ids)
