import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import RANDOM_SEED, RUN_METADATA_DIR
from src.operations.configuration import config_hash, load_platform_config, save_config_snapshot


def set_deterministic_seed(seed: int = RANDOM_SEED) -> None:
    # Deterministic seeds make model training and simulations easier to compare.
    # They do not make markets predictable; they make our own software behavior
    # reproducible enough to debug.
    np.random.seed(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def create_run_metadata(
    workflow: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_platform_config()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{workflow}_{timestamp}"
    snapshot_path = save_config_snapshot(run_id, config)
    metadata = {
        "run_id": run_id,
        "workflow": workflow,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "random_seed": RANDOM_SEED,
        "config_hash": config_hash(config),
        "config_snapshot_path": snapshot_path,
        "extra": extra or {},
    }
    output = Path(RUN_METADATA_DIR) / f"{run_id}_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    metadata["metadata_path"] = str(output)
    return metadata
