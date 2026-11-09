"""Back up, verify and restore the simulated paper account (local files only).

    python scripts/paper_backup.py backup                 # -> backups/paper/paper-backup-<UTC>.tar.gz
    python scripts/paper_backup.py backup --out FILE      # refuses an existing FILE without --force
    python scripts/paper_backup.py verify FILE            # integrity check (SHA-256 per file)
    python scripts/paper_backup.py restore FILE           # refuses to overwrite existing state
    python scripts/paper_backup.py restore FILE --force   # safety backup of current state, then replace
    python scripts/paper_backup.py list

Covers results/paper/, data/paper/ and the top-level results/simulation/ broker state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.operations.paper_backup import (  # noqa: E402
    BackupError,
    create_backup,
    list_backups,
    restore_backup,
    verify_backup,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backup", help="Write a verified .tar.gz of the paper state.")
    b.add_argument("--out", default=None, help="Archive path (default backups/paper/paper-backup-<UTC>.tar.gz).")
    b.add_argument("--force", action="store_true", help="Overwrite --out if it exists.")
    v = sub.add_parser("verify", help="Check an archive against its manifest.")
    v.add_argument("archive")
    r = sub.add_parser("restore", help="Verify, then restore an archive.")
    r.add_argument("archive")
    r.add_argument("--force", action="store_true",
                   help="Replace existing paper state (a safety backup of it is written first).")
    sub.add_parser("list", help="List archives under backups/paper/.")
    args = parser.parse_args(argv)

    try:
        if args.command == "backup":
            out, manifest = create_backup(args.out, force=args.force)
            print(f"Backup written and verified: {out} ({len(manifest['files'])} files)")
        elif args.command == "verify":
            manifest = verify_backup(args.archive)
            print(f"OK: {args.archive} ({len(manifest['files'])} files, created {manifest['created_at']}, "
                  f"commit {manifest.get('git_commit') or 'unknown'})")
        elif args.command == "restore":
            result = restore_backup(args.archive, force=args.force)
            print(f"Restored {result['restored_files']} files from {args.archive} "
                  f"(backup created {result['created_at']}).")
            if result["safety_backup"]:
                print(f"Previous state saved to {result['safety_backup']}")
            print("Next: make paper-status, then a paper run reconciles against the restored state.")
        else:
            backups = list_backups()
            print("\n".join(str(p) for p in backups) if backups else "No backups under backups/paper/.")
    except BackupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
