from __future__ import annotations

from fastapi import HTTPException
from openenv.core.env_server.http_server import create_app

from env.models import (
    Action,
    BaselineRequest,
    BaselineResponse,
    GraderRequest,
    GraderResponse,
    Observation,
    TaskPreview,
)
from server.netra_environment import NetraDroneSurveillanceServer

_SHARED_SERVER = NetraDroneSurveillanceServer()

def _get_server() -> NetraDroneSurveillanceServer:
    return _SHARED_SERVER

base_app = create_app(
    _get_server,
    Action,
    Observation,
    env_name="netra-drone-surveillance",
    max_concurrent_envs=1,
)

app = base_app
app.title = "Netra Drone Surveillance API"

def _list_tasks() -> list[TaskPreview]:
    return _get_server().tasks()

def _grade_plan(request: GraderRequest) -> GraderResponse:
    try:
        return _get_server().grade(
            request.task_id,
            [action.value for action in request.action_plan],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

def _run_baseline(request: BaselineRequest | None = None) -> BaselineResponse:
    payload = request or BaselineRequest()
    try:
        return _get_server().baseline(
            seed=payload.seed,
            task_ids=payload.task_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Netra Drone Surveillance API is running",
        "docs": "/docs",
        "api_base": "/api",
    }

@app.get("/tasks", response_model=list[TaskPreview])
def list_tasks() -> list[TaskPreview]:
    return _list_tasks()

@app.post("/grader", response_model=GraderResponse)
def grade_plan_endpoint(request: GraderRequest) -> GraderResponse:
    return _grade_plan(request)

@app.post("/baseline", response_model=BaselineResponse)
def run_baseline_endpoint(request: BaselineRequest | None = None) -> BaselineResponse:
    return _run_baseline(request)

@app.get("/api/tasks", response_model=list[TaskPreview])
def list_tasks_api() -> list[TaskPreview]:
    return _list_tasks()

@app.post("/api/grader", response_model=GraderResponse)
def grade_plan_endpoint_api(request: GraderRequest) -> GraderResponse:
    return _grade_plan(request)

@app.post("/api/baseline", response_model=BaselineResponse)
def run_baseline_endpoint_api(
    request: BaselineRequest | None = None,
) -> BaselineResponse:
    return _run_baseline(request)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}

@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "healthy"}

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
