import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import CONFIG_SNAPSHOT_DIR, LOCAL_PLATFORM_CONFIG_PATH, PLATFORM_CONFIG_PATH


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_platform_config(
    default_path: str = PLATFORM_CONFIG_PATH,
    local_path: str = LOCAL_PLATFORM_CONFIG_PATH,
) -> dict[str, Any]:
    default = json.loads(Path(default_path).read_text())
    local = Path(local_path)
    if local.exists():
        # Local overrides let an operator change costs or thresholds without
        # editing code. Keeping overrides separate improves deployment hygiene.
        return deep_update(default, json.loads(local.read_text()))
    return default


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def save_config_snapshot(
    run_id: str,
    config: dict[str, Any] | None = None,
    output_dir: str = CONFIG_SNAPSHOT_DIR,
) -> str:
    # Reproducibility matters because research and paper-trading decisions must
    # be explainable later. A config snapshot answers "what settings produced
    # this run?" without relying on memory or mutable source files.
    config = config or load_platform_config()
    snapshot = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_hash": config_hash(config),
        "config": config,
    }
    path = Path(output_dir) / f"{run_id}_config_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    return str(path)
