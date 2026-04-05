"""Deploy the Netra OpenEnv project to a Hugging Face Docker Space."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterable

import requests
from huggingface_hub import HfApi


DEFAULT_ALLOW_PATTERNS = [
    "README.md",
    "Dockerfile",
    ".dockerignore",
    "openenv.yaml",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "environment.py",
    "tasks.py",
    "api/**",
    "baseline/**",
    "env/**",
    "scripts/**",
    "server/**",
]

TERMINAL_FAILURE_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "PAUSED", "NO_APP_FILE"}


def _resolve_token() -> str:
    """Return a Hugging Face token from the environment or raise."""

    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, "
            "or HUGGING_FACE_HUB_TOKEN before running this script."
        )
    return token


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Build the CLI for Space deployment."""

    parser = argparse.ArgumentParser(
        description="Create, upload, and optionally monitor a Hugging Face Docker Space."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Space repo id, for example 'username/netra-drone-surveillance-environment'.",
    )
    parser.add_argument(
        "--folder",
        default=".",
        help="Project folder to upload. Defaults to the current directory.",
    )
    parser.add_argument(
        "--message",
        default="Deploy Netra Drone Surveillance OpenEnv environment",
        help="Commit message used for the upload.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart the Space after upload to force a fresh remote build.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the Space to leave BUILDING/APP_STARTING and report the final stage.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Maximum wait time in seconds when --wait is used. Defaults to 1200.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Polling interval in seconds when --wait is used. Defaults to 30.",
    )
    return parser.parse_args(argv)


def _wait_for_space(api: HfApi, repo_id: str, timeout: int, poll_interval: int) -> dict:
    """Wait for a Space to finish building and return the final runtime payload."""

    deadline = time.time() + timeout
    last_stage: str | None = None

    while True:
        runtime = api.get_space_runtime(repo_id=repo_id)
        stage = getattr(runtime, "stage", "UNKNOWN")
        if stage != last_stage:
            print(f"[space] stage={stage}")
            last_stage = stage

        if stage == "RUNNING":
            return runtime.raw
        if stage in TERMINAL_FAILURE_STAGES:
            return runtime.raw
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for Space '{repo_id}' after {timeout} seconds."
            )
        time.sleep(poll_interval)


def _probe_space(repo_id: str) -> None:
    """Best-effort probe of the public health endpoint after the Space is running."""

    slug = repo_id.split("/", 1)[1].replace("_", "-")
    owner = repo_id.split("/", 1)[0].lower()
    base_url = f"https://{owner}-{slug}.hf.space"

    for path in ("/health", "/metadata", "/schema"):
        url = base_url + path
        try:
            response = requests.get(url, timeout=30)
            print(f"[probe] {url} -> {response.status_code}")
        except requests.RequestException as exc:
            print(f"[probe] {url} -> {type(exc).__name__}: {exc}")


def main(argv: Iterable[str] | None = None) -> int:
    """Create or update a Docker Space and optionally monitor its build."""

    args = _parse_args(argv)
    token = _resolve_token()
    api = HfApi(token=token)

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print(f"[deploy] ensured Space exists: {args.repo_id}")

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=args.folder,
        path_in_repo=".",
        allow_patterns=DEFAULT_ALLOW_PATTERNS,
        commit_message=args.message,
    )
    print(f"[deploy] uploaded project files to {args.repo_id}")

    if args.restart:
        api.restart_space(repo_id=args.repo_id)
        print(f"[deploy] restarted Space build for {args.repo_id}")

    space_url = f"https://huggingface.co/spaces/{args.repo_id}"
    print(f"[deploy] Space page: {space_url}")

    if args.wait:
        runtime = _wait_for_space(
            api=api,
            repo_id=args.repo_id,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
        )
        print(f"[deploy] final runtime: {runtime}")
        if runtime.get("stage") == "RUNNING":
            _probe_space(args.repo_id)
            return 0
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
