"""Backup / verify / restore of the simulated paper account (local files only).

A backup is a ``.tar.gz`` holding ``BACKUP_MANIFEST.json`` plus every file of:

- ``results/paper/``       run snapshots, ledger, journal, alerts, status
- ``data/paper/``          paper market data + its manifest
- ``results/simulation/``  top-level simulated broker state files (broker_state.json,
                           trade history, snapshots ...). Needed for continuity: the
                           next run reconciles its starting state against the last
                           run's end state, so restoring results/paper/ without the
                           broker state would be flagged as a mismatch.

The manifest records the SHA-256 and size of each file. ``verify`` checks the
archive against it (and refuses absolute paths, ``..``, links and members that
are not listed). ``restore`` verifies first, refuses to overwrite existing state
unless ``force=True`` (then it saves a safety backup of the current state
first), extracts into a staging folder, re-checks every hash, and only then
swaps the folders into place.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

BACKUP_FORMAT = 1
MANIFEST_NAME = "BACKUP_MANIFEST.json"
DEFAULT_BACKUP_DIR = "backups/paper"


class BackupError(RuntimeError):
    """The archive is invalid, or a restore would overwrite state without --force."""


def default_sources() -> list[str]:
    """Directories (trailing ``/``) and files that make up the paper account."""
    from src.config import PAPER_OPS_DIR
    from src.data.paths import DataPaths

    return [f"{PAPER_OPS_DIR}/", f"{DataPaths.paper().root}/", "results/simulation/*"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _skip(rel: PurePosixPath) -> bool:
    # In-flight staging folders and temp files are not part of a consistent state.
    return any(p.startswith((".staging-", ".replaced-")) or p.endswith((".tmp", ".restore")) for p in rel.parts)


def collect_files(sources: list[str], root: str | Path = ".") -> dict[str, Path]:
    """Map archive name (posix, relative to root) -> file path for every source."""
    root = Path(root)
    files: dict[str, Path] = {}
    for source in sources:
        if source.endswith("/*"):
            base = root / source[:-2]
            candidates = sorted(p for p in base.glob("*") if p.is_file()) if base.exists() else []
        elif source.endswith("/"):
            base = root / source.rstrip("/")
            candidates = sorted(p for p in base.rglob("*") if p.is_file()) if base.exists() else []
        else:
            candidates = [root / source] if (root / source).is_file() else []
        for path in candidates:
            try:
                rel = PurePosixPath(path.relative_to(root).as_posix())
            except ValueError as exc:
                raise BackupError(f"{path} is outside {root}; back up a paper dir inside the repo") from exc
            if not _skip(rel) and not path.is_symlink():
                files[str(rel)] = path
    return files


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def default_backup_path(root: str | Path = ".") -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(root) / DEFAULT_BACKUP_DIR / f"paper-backup-{stamp}.tar.gz"


def create_backup(
    out: str | Path | None = None,
    sources: list[str] | None = None,
    root: str | Path = ".",
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    root = Path(root)
    sources = sources or default_sources()
    out = Path(out) if out else default_backup_path(root)
    if out.exists() and not force:
        raise BackupError(f"{out} already exists; pass --force to overwrite it")
    files = collect_files(sources, root)
    if not files:
        raise BackupError(f"nothing to back up under {', '.join(sources)}")
    manifest = {
        "format": BACKUP_FORMAT,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(root),
        "sources": sources,
        "files": {name: {"sha256": _sha256(p), "size": p.stat().st_size} for name, p in files.items()},
        "note": "Simulated paper-trading state only; restore with scripts/paper_backup.py restore.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{uuid.uuid4().hex}.tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(payload)
        info.mtime = int(datetime.now(UTC).timestamp())
        tar.addfile(info, io.BytesIO(payload))
        for name, path in files.items():
            tar.add(path, arcname=name, recursive=False)
    verify_backup(tmp)
    os.replace(tmp, out)
    return out, manifest


def _safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and name not in ("", ".")


def _covered(name: str, sources: list[str]) -> bool:
    for source in sources:
        if source.endswith("/*") and str(PurePosixPath(name).parent) == source[:-2]:
            return True
        if source.endswith("/") and name.startswith(source):
            return True
        if name == source:
            return True
    return False


def verify_backup(archive: str | Path) -> dict[str, Any]:
    """Return the manifest if every member is safe, listed and matches its hash."""
    archive = Path(archive)
    if not archive.is_file():
        raise BackupError(f"{archive} not found")
    try:
        tar = tarfile.open(archive, "r:gz")
    except (tarfile.TarError, OSError) as exc:
        raise BackupError(f"{archive} is not a readable .tar.gz: {exc}") from exc
    with tar:
        try:
            member = tar.getmember(MANIFEST_NAME)
            manifest = json.loads(tar.extractfile(member).read())
        except (KeyError, json.JSONDecodeError, AttributeError) as exc:
            raise BackupError(f"{archive} has no valid {MANIFEST_NAME}") from exc
        if manifest.get("format") != BACKUP_FORMAT:
            raise BackupError(f"unsupported backup format {manifest.get('format')!r}")
        recorded = manifest.get("files") or {}
        sources = list(manifest.get("sources") or [])
        seen = set()
        for m in tar.getmembers():
            if m.name == MANIFEST_NAME:
                continue
            if not m.isfile():
                raise BackupError(f"unexpected non-file member {m.name!r}")
            if not _safe_name(m.name) or not _covered(m.name, sources):
                raise BackupError(f"unsafe or out-of-scope member path {m.name!r}")
            if m.name not in recorded:
                raise BackupError(f"member {m.name!r} is not listed in the manifest")
            digest = hashlib.sha256(tar.extractfile(m).read()).hexdigest()
            if digest != recorded[m.name]["sha256"]:
                raise BackupError(f"hash mismatch for {m.name!r}: archive is corrupt or was modified")
            seen.add(m.name)
        missing = sorted(set(recorded) - seen)
        if missing:
            raise BackupError(f"manifest lists files missing from the archive: {', '.join(missing[:5])}")
    return manifest


def _existing_state(sources: list[str], root: Path) -> list[str]:
    return sorted(collect_files(sources, root))


def restore_backup(
    archive: str | Path,
    root: str | Path = ".",
    force: bool = False,
    safety_backup_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify, then replace the paper state with the archive's content.

    Without ``force`` any existing file under the archive's sources aborts the
    restore. With ``force`` the current state is first saved as a safety backup.
    """
    root = Path(root)
    manifest = verify_backup(archive)
    sources = list(manifest["sources"])
    existing = _existing_state(sources, root)
    safety = None
    if existing:
        if not force:
            raise BackupError(
                f"refusing to overwrite {len(existing)} existing paper-state file(s) "
                f"(e.g. {existing[0]}); pass --force to replace them (a safety backup is taken first)"
            )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target_dir = Path(safety_backup_dir) if safety_backup_dir else root / DEFAULT_BACKUP_DIR
        safety, _ = create_backup(target_dir / f"pre-restore-{stamp}-{uuid.uuid4().hex[:6]}.tar.gz",
                                  sources=sources, root=root)

    staging = root / f".paper-restore-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name == MANIFEST_NAME:
                    continue
                dest = staging / m.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
        for name, meta in manifest["files"].items():
            if _sha256(staging / name) != meta["sha256"]:
                raise BackupError(f"hash mismatch after extraction for {name!r}; nothing was changed")

        # Swap into place: whole directories for "dir/" sources, top-level files for "dir/*".
        for source in sources:
            if source.endswith("/"):
                target = root / source.rstrip("/")
                incoming = staging / source.rstrip("/")
                if target.exists():
                    shutil.rmtree(target)
                if incoming.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(incoming, target)
            elif source.endswith("/*"):
                base = root / source[:-2]
                for old in (sorted(p for p in base.glob("*") if p.is_file()) if base.exists() else []):
                    old.unlink()
                incoming = staging / source[:-2]
                if incoming.exists():
                    base.mkdir(parents=True, exist_ok=True)
                    for f in sorted(p for p in incoming.glob("*") if p.is_file()):
                        os.replace(f, base / f.name)
            else:
                incoming = staging / source
                target = root / source
                if target.exists():
                    target.unlink()
                if incoming.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(incoming, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"restored_files": len(manifest["files"]), "created_at": manifest.get("created_at"),
            "git_commit": manifest.get("git_commit"), "safety_backup": str(safety) if safety else None}


def list_backups(root: str | Path = ".") -> list[Path]:
    base = Path(root) / DEFAULT_BACKUP_DIR
    return sorted(base.glob("*.tar.gz")) if base.exists() else []
