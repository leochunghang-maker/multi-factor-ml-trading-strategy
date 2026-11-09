"""Verify (or fix) numbers quoted in README.md and reports/*.md against result files.

    python src/check_doc_metrics.py          # exit 1 on any mismatch
    python src/check_doc_metrics.py --fix    # rewrite mismatched marked values
    python src/check_doc_metrics.py --list   # print every checkable key and value

Quote a number in a doc as ``<!--m:KEY-->VALUE<!--/m-->`` (see src/reporting/doc_metrics.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.reporting.doc_metrics import (  # noqa: E402
    DEFAULT_DOCUMENTS,
    MARKER,
    check_documents,
    expected_metrics,
    fix_documents,
)
from src.data.required import run_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="Rewrite mismatched values in place.")
    parser.add_argument("--list", action="store_true", help="Print all keys and expected values.")
    args = parser.parse_args(argv)

    expected = expected_metrics()
    if args.list:
        for key in sorted(expected):
            print(f"{key:40s} {expected[key]}")
        return 0
    if args.fix:
        print(f"Updated {fix_documents(expected=expected)} marked value(s).")
    issues = check_documents(expected=expected)
    markers = sum(len(MARKER.findall(Path(doc).read_text())) for doc in DEFAULT_DOCUMENTS)
    for issue in issues:
        print(issue)
    print(f"Doc metrics: {markers} marked values in {len(DEFAULT_DOCUMENTS)} documents, {len(issues)} issue(s).")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(run_cli(main))
