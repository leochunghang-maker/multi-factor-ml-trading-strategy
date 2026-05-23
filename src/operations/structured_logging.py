import json
import logging as standard_logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import STRUCTURED_LOG_PATH


def configure_text_logging(level: int = standard_logging.INFO) -> None:
    standard_logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def log_event(
    event_type: str,
    message: str,
    level: str = "INFO",
    run_id: str | None = None,
    path: str = STRUCTURED_LOG_PATH,
    **fields: Any,
) -> None:
    # Monitoring matters in quant systems because silent failures can look like
    # valid returns until capital is exposed. JSONL logs make events searchable
    # by run id, severity, module, or safety check.
    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "level": level,
        "event_type": event_type,
        "message": message,
        "run_id": run_id,
        **fields,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as file:
        file.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def read_recent_events(path: str = STRUCTURED_LOG_PATH, limit: int = 50) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    lines = log_path.read_text().splitlines()[-limit:]
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
