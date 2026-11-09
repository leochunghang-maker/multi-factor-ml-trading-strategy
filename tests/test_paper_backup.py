"""Backup / verify / restore of the paper account (temp dirs only)."""

import io
import json
import tarfile
from pathlib import Path

import pytest

from src.operations.paper_backup import (
    MANIFEST_NAME,
    BackupError,
    collect_files,
    create_backup,
    restore_backup,
    verify_backup,
)

SOURCES = ["results/paper/", "data/paper/", "results/simulation/*"]


def _make_state(root: Path, nav: str = "100000") -> dict[str, bytes]:
    files = {
        "results/paper/ledger.csv": f"signal_date,nav_end\n2026-11-06,{nav}\n".encode(),
        "results/paper/runs/2026-11-06/run.json": b'{"status": "COMPLETED"}',
        "data/paper/price_data.csv": b"Date,AAA\n2026-11-06,10\n",
        "results/simulation/broker_state.json": f'{{"cash": {nav}}}'.encode(),
        "results/simulation/multi_day/keep.csv": b"not part of the paper account\n",
        "results/paper/runs/.staging-2026-11-07-x/run.json": b"in flight",
    }
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return files


def _snapshot(root: Path) -> dict[str, bytes]:
    return {name: path.read_bytes() for name, path in collect_files(SOURCES, root).items()}


def test_backup_round_trip_into_empty_root(tmp_path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_state(src)
    archive, manifest = create_backup(tmp_path / "b.tar.gz", sources=SOURCES, root=src)
    assert set(manifest["files"]) == {
        "results/paper/ledger.csv", "results/paper/runs/2026-11-06/run.json",
        "data/paper/price_data.csv", "results/simulation/broker_state.json",
    }  # staging folders and nested multi_day/ are excluded
    assert verify_backup(archive)["files"] == manifest["files"]
    dst.mkdir()
    result = restore_backup(archive, root=dst)
    assert result["restored_files"] == 4 and result["safety_backup"] is None
    assert _snapshot(dst) == _snapshot(src)


def test_backup_refuses_existing_output_without_force(tmp_path) -> None:
    _make_state(tmp_path)
    out = tmp_path / "b.tar.gz"
    create_backup(out, sources=SOURCES, root=tmp_path)
    with pytest.raises(BackupError, match="already exists"):
        create_backup(out, sources=SOURCES, root=tmp_path)
    create_backup(out, sources=SOURCES, root=tmp_path, force=True)


def test_restore_refuses_overwrite_then_force_saves_safety_backup(tmp_path) -> None:
    _make_state(tmp_path, nav="100000")
    archive, _ = create_backup(tmp_path / "old.tar.gz", sources=SOURCES, root=tmp_path)
    old = _snapshot(tmp_path)
    _make_state(tmp_path, nav="95000")
    (tmp_path / "results/paper/runs/2026-11-07").mkdir()
    (tmp_path / "results/paper/runs/2026-11-07/run.json").write_text("{}")
    newer = _snapshot(tmp_path)

    with pytest.raises(BackupError, match="refusing to overwrite"):
        restore_backup(archive, root=tmp_path)
    assert _snapshot(tmp_path) == newer  # untouched

    result = restore_backup(archive, root=tmp_path, force=True, safety_backup_dir=tmp_path / "safety")
    assert _snapshot(tmp_path) == old
    assert not (tmp_path / "results/paper/runs/2026-11-07").exists()
    assert (tmp_path / "results/simulation/multi_day/keep.csv").exists()  # outside the account
    safety = Path(result["safety_backup"])
    assert set(verify_backup(safety)["files"]) == set(newer)


def _rewrite_archive(src: Path, dst: Path, mutate) -> None:
    with tarfile.open(src, "r:gz") as tin, tarfile.open(dst, "w:gz") as tout:
        for m in tin.getmembers():
            data = tin.extractfile(m).read()
            m, data = mutate(m, data)
            if m is None:
                continue
            m.size = len(data)
            tout.addfile(m, io.BytesIO(data))


def test_verify_detects_tampering_missing_files_and_unsafe_paths(tmp_path) -> None:
    _make_state(tmp_path)
    archive, _ = create_backup(tmp_path / "b.tar.gz", sources=SOURCES, root=tmp_path)

    def tamper(m, data):
        return m, (data + b"x" if m.name == "results/simulation/broker_state.json" else data)

    def drop(m, data):
        return (None, data) if m.name == "data/paper/price_data.csv" else (m, data)

    def traverse(m, data):
        if m.name == "results/paper/ledger.csv":
            m.name = "results/paper/../../etc/ledger.csv"
        return m, data

    def extra(m, data):
        if m.name == MANIFEST_NAME:
            manifest = json.loads(data)
            manifest["files"]["../evil"] = {"sha256": "0", "size": 1}
            return m, json.dumps(manifest).encode()
        return m, data

    for mutate, pattern in [(tamper, "hash mismatch"), (drop, "missing from the archive"),
                            (traverse, "unsafe or out-of-scope"), (extra, "missing from the archive")]:
        bad = tmp_path / f"bad-{mutate.__name__}.tar.gz"
        _rewrite_archive(archive, bad, mutate)
        with pytest.raises(BackupError, match=pattern):
            verify_backup(bad)
        with pytest.raises(BackupError):
            restore_backup(bad, root=tmp_path / "fresh")
        assert not (tmp_path / "fresh" / "results").exists()

    (tmp_path / "junk.tar.gz").write_bytes(b"not an archive")
    with pytest.raises(BackupError, match="not a readable"):
        verify_backup(tmp_path / "junk.tar.gz")


def test_backup_with_nothing_to_save_fails(tmp_path) -> None:
    with pytest.raises(BackupError, match="nothing to back up"):
        create_backup(tmp_path / "b.tar.gz", sources=SOURCES, root=tmp_path)
