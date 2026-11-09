"""Single-instance lock file for scheduled paper runs (prevents overlapping runs).

The lock is a JSON file created with ``O_CREAT | O_EXCL`` (atomic on local
filesystems) holding pid, hostname and start time. A lock is **stale** and is
taken over (with a log line) when either

- it was written on this host and its pid no longer exists, or
- it is older than ``stale_after_seconds`` (a hung or killed run on any host),
- or the file is unreadable/corrupt.

Otherwise ``acquire`` raises ``LockHeld`` and the caller should exit without
doing anything.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


UNREADABLE_GRACE_SECONDS = 30


class LockHeld(RuntimeError):
    """Another live run holds the lock."""


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by another user
        return True
    except OSError:
        return False
    return True


@dataclass
class RunLock:
    path: Path
    stale_after_seconds: float = 6 * 3600
    log: Callable[[str], None] = print
    owner: dict[str, Any] = field(default_factory=dict)
    acquired: bool = False

    def read(self) -> dict[str, Any] | None:
        try:
            return json.loads(Path(self.path).read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return {}

    def stale_reason(self, info: dict[str, Any] | None, now: float | None = None) -> str | None:
        """Why an existing lock may be taken over, or None if it is live."""
        if info is None:
            return None
        now = time.time() if now is None else now
        if not info or "pid" not in info:
            try:
                written = Path(self.path).stat().st_mtime
            except FileNotFoundError:
                return None
            # A just-created lock may not be written yet; only take over old unreadable ones.
            if now - written < UNREADABLE_GRACE_SECONDS:
                return None
            return "lock file unreadable or incomplete"
        age = now - float(info.get("started_epoch", 0))
        if age > self.stale_after_seconds:
            return f"lock is {age / 3600:.1f} h old (limit {self.stale_after_seconds / 3600:.1f} h)"
        if info.get("hostname") == socket.gethostname() and not pid_alive(int(info["pid"])):
            return f"pid {info['pid']} on this host is not running"
        return None

    def acquire(self) -> "RunLock":
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info = self.read()
                reason = self.stale_reason(info)
                if reason is None:
                    if info is None:  # released between our open and read: retry once
                        continue
                    info = info or {}
                    raise LockHeld(
                        f"another paper run holds {path} (pid {info.get('pid')} on {info.get('hostname')}, "
                        f"started {info.get('started_at')}); not starting a second run"
                    )
                self.log(f"Removing stale lock {path}: {reason}")
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            self.owner = {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "started_epoch": time.time(),
            }
            with os.fdopen(fd, "w") as fh:
                json.dump(self.owner, fh)
            self.acquired = True
            return self
        raise LockHeld(f"could not acquire {path}")

    def release(self) -> None:
        if not self.acquired:
            return
        info = self.read()
        # Only remove our own lock (a stale-takeover by another run must not be undone).
        if info and info.get("pid") == self.owner.get("pid") and info.get("started_epoch") == self.owner.get(
            "started_epoch"
        ):
            Path(self.path).unlink(missing_ok=True)
        self.acquired = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
