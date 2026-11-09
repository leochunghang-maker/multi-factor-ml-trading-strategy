"""requirements.lock (pinned, hashed) must agree with requirements.txt (direct deps).

Regenerate the lock after editing requirements.txt:
    uv pip compile requirements.txt --python-version 3.12 --universal --generate-hashes -o requirements.lock
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        match = PIN.match(line.strip())
        if match:
            pins[_normalize(match.group(1))] = match.group(2)
    return pins


def test_every_direct_dependency_is_locked_at_the_same_version() -> None:
    direct = _pins(ROOT / "requirements.txt")
    locked = _pins(ROOT / "requirements.lock")
    assert direct, "requirements.txt has no pinned requirements"
    mismatched = {name: (version, locked.get(name)) for name, version in direct.items() if locked.get(name) != version}
    assert not mismatched, (
        f"requirements.lock is out of date for {mismatched}; regenerate it with "
        "`uv pip compile requirements.txt --python-version 3.12 --universal --generate-hashes -o requirements.lock`"
    )


def test_every_locked_package_has_a_hash() -> None:
    text = (ROOT / "requirements.lock").read_text()
    blocks = re.split(r"\n(?=[A-Za-z0-9])", text.split("\n", 2)[2])
    unhashed = [block.split()[0] for block in blocks if block.strip() and "--hash=sha256:" not in block]
    assert not unhashed, f"locked packages without hashes (pip --require-hashes would fail): {unhashed}"
