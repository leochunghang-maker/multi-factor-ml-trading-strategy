"""Print and write the simulated paper account status (results/paper/status.md).

    python scripts/paper_status.py            # print the summary, refresh status.md/status.json
    python scripts/paper_status.py --strict   # exit 1 if any active alert is ERROR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.operations.paper_ops import PaperOpsPaths  # noqa: E402
from src.operations.paper_status import render_paper_status, write_paper_status  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="Exit 1 when an active alert is ERROR.")
    args = parser.parse_args(argv)
    paths = PaperOpsPaths.default()
    status = write_paper_status(paths)
    print(render_paper_status(status))
    print(f"Written to {paths.status_md} and {paths.status_json}")
    return 1 if args.strict and status["overall_status"] == "ERROR" else 0


if __name__ == "__main__":
    sys.exit(main())
