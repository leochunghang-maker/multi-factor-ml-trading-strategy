import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import REPORTS_DIR, SUMMARY_REPORT_CSV_PATH, SUMMARY_REPORT_MARKDOWN_PATH
from src.reporting.summary import generate_summary_report


def main() -> None:
    report = generate_summary_report()

    print("Institutional Risk Summary")
    print()
    print(report.to_string(index=False))
    print()
    print(f"Charts saved to {REPORTS_DIR}/")
    print(f"CSV report saved to {SUMMARY_REPORT_CSV_PATH}")
    print(f"Markdown report saved to {SUMMARY_REPORT_MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
