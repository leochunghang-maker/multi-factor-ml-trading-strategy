"""Write the weekly paper review (results/paper/reviews/weekly-<YYYY>-W<ww>.md/.json).

    python scripts/paper_weekly_review.py                    # current ISO week (UTC)
    python scripts/paper_weekly_review.py --week-of 2026-11-06
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.operations.paper_review import render_weekly_review, write_weekly_review  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week-of", default=None, help="Any date inside the week (default today, UTC).")
    args = parser.parse_args(argv)
    path, review = write_weekly_review(week_of=args.week_of)
    print(render_weekly_review(review))
    print(f"Written to {path} and {path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
