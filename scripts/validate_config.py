"""Validate the local OpenEnv environment using the installed official CLI package."""

from __future__ import annotations

import json
from pathlib import Path

from openenv.cli._validation import (
    build_local_validation_json_report,
    get_deployment_modes,
    validate_multi_mode_deployment,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    is_valid, issues = validate_multi_mode_deployment(root)
    report = build_local_validation_json_report(
        env_name=root.name,
        env_path=root,
        is_valid=is_valid,
        issues=issues,
        deployment_modes=get_deployment_modes(root),
    )
    print(json.dumps(report, indent=2))
    return 0 if is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
