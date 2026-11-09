"""Clear one-line errors for missing input files.

``require_file(path)`` raises :class:`MissingInputError` naming the missing
file and the command that creates it, instead of a pandas traceback.
``run_cli(main)`` wraps a script's ``main`` so that error prints as a single
``ERROR:`` line on stderr and exits with code 2; every other exception is
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

MISSING_INPUT_EXIT_CODE = 2

# Checked-in research inputs: restore the snapshot rather than re-download
# (a plain `python src/data_loader.py` rewrites the research data).
_CHECKED_IN = "git checkout -- {path}"
_RESEARCH_PRODUCERS: dict[str, str] = {
    "price_data.csv": _CHECKED_IN,
    "volume_data.csv": _CHECKED_IN,
    "benchmark_prices.csv": _CHECKED_IN,
    "daily_returns.csv": "python src/factors.py",
    "momentum_12m.csv": "python src/factors.py",
    "mean_reversion_1m.csv": "python src/factors.py",
    "forward_returns_1m.csv": "python src/factors.py",
    "ml_dataset.csv": "python src/ml_dataset.py",
    "ml_predictions.csv": "python src/train_ml_model.py",
    "xgboost_predictions.csv": "python src/train_xgboost_model.py",
    "multi_factor_results.csv": "python src/multi_factor_backtest.py",
    "walk_forward_period_returns.csv": "python src/walk_forward_ml_backtest.py",
    "significance_report.csv": "python src/run_significance_report.py --cached-walk-forward",
    "benchmark_relative_report.csv": "python src/run_significance_report.py --cached-walk-forward",
    "latest_signals.csv": "make signals",
}
# The paper workflow rebuilds features itself; only the downloaded inputs can be missing.
_PAPER_PRODUCERS: dict[str, str] = {
    "price_data.csv": "python src/data_loader.py --paper",
    "volume_data.csv": "python src/data_loader.py --paper",
    "benchmark_prices.csv": "python src/data_loader.py --paper",
}


class MissingInputError(FileNotFoundError):
    """A required input file does not exist; the message says how to create it."""


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def producer_command(path: str | Path) -> str | None:
    """The command that creates ``path``, or None when it is not a known artifact."""
    path = Path(path)
    paper = "paper" in path.parts[:-1]
    command = (_PAPER_PRODUCERS if paper else _RESEARCH_PRODUCERS).get(path.name)
    if command is None and paper:
        command = "python scripts/run_daily_paper_trading.py"
    return command.format(path=_display(path)) if command else None


def missing_input_message(path: str | Path, command: str | None = None) -> str:
    shown = _display(Path(path))
    command = command or producer_command(path)
    if command:
        return f"Missing input file {shown}; create it with `{command}`."
    return f"Missing input file {shown}; check the path (no known command creates it)."


def require_file(path: str | Path, command: str | None = None) -> Path:
    """Return ``Path(path)`` or raise MissingInputError naming the creating command."""
    path = Path(path)
    if not path.is_file():
        raise MissingInputError(missing_input_message(path, command))
    return path


def run_cli(main: Callable[[], Any]) -> Any:
    """Call ``main()``; a MissingInputError becomes one stderr line and exit code 2."""
    try:
        return main()
    except MissingInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(MISSING_INPUT_EXIT_CODE) from None
