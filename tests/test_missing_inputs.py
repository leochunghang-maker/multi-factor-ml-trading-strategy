"""Missing input files fail with one line naming the file and the command that creates it."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.data.io import load_ml_dataset, load_price_data
from src.data.required import (
    MISSING_INPUT_EXIT_CODE,
    MissingInputError,
    producer_command,
    require_file,
    run_cli,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("path", "command"),
    [
        ("data/price_data.csv", "git checkout -- data/price_data.csv"),
        ("data/momentum_12m.csv", "python src/factors.py"),
        ("data/ml_dataset.csv", "python src/ml_dataset.py"),
        ("results/multi_factor_results.csv", "python src/multi_factor_backtest.py"),
        ("results/significance_report.csv", "python src/run_significance_report.py --cached-walk-forward"),
        ("results/live/latest_signals.csv", "make signals"),
        ("data/paper/price_data.csv", "python src/data_loader.py --paper"),
        ("data/paper/ml_dataset.csv", "python scripts/run_daily_paper_trading.py"),
    ],
)
def test_producer_command_for_known_artifacts(path: str, command: str) -> None:
    assert producer_command(path) == command


def test_require_file_message_is_one_line_with_file_and_command(tmp_path) -> None:
    missing = tmp_path / "data" / "ml_dataset.csv"
    with pytest.raises(MissingInputError) as info:
        require_file(missing)
    message = str(info.value)
    assert "\n" not in message
    assert str(missing) in message or "data/ml_dataset.csv" in message
    assert "`python src/ml_dataset.py`" in message
    assert isinstance(info.value, FileNotFoundError)  # existing except FileNotFoundError still works


def test_require_file_explicit_command_and_unknown_file(tmp_path) -> None:
    with pytest.raises(MissingInputError, match="create it with `make paper`"):
        require_file(tmp_path / "x.csv", command="make paper")
    with pytest.raises(MissingInputError, match="no known command creates it"):
        require_file(tmp_path / "unknown.csv")
    present = tmp_path / "present.csv"
    present.write_text("a\n1\n")
    assert require_file(present) == present


def test_loaders_raise_missing_input_error(tmp_path) -> None:
    with pytest.raises(MissingInputError, match="price_data.csv"):
        load_price_data(str(tmp_path / "price_data.csv"))
    with pytest.raises(MissingInputError, match="ml_dataset.py"):
        load_ml_dataset(str(tmp_path / "ml_dataset.csv"))


def test_run_cli_prints_one_line_and_exits_2(capsys) -> None:
    def main() -> None:
        require_file("results/definitely_missing/multi_factor_results.csv")

    with pytest.raises(SystemExit) as info:
        run_cli(main)
    assert info.value.code == MISSING_INPUT_EXIT_CODE
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("ERROR: Missing input file ")
    assert "python src/multi_factor_backtest.py" in err


def test_run_cli_passes_through_results_and_other_errors() -> None:
    assert run_cli(lambda: 0) == 0
    with pytest.raises(ValueError):
        run_cli(lambda: (_ for _ in ()).throw(ValueError("boom")))


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("src/factors.py", "data/price_data.csv; create it with `git checkout -- data/price_data.csv`"),
        ("src/check_doc_metrics.py", "results/multi_factor_results.csv; create it with "
                                     "`python src/multi_factor_backtest.py`"),
    ],
)
def test_scripts_report_missing_input_without_traceback(tmp_path, script: str, expected: str) -> None:
    # Run from an empty directory so the repo-relative inputs do not exist.
    completed = subprocess.run(
        [sys.executable, str(ROOT / script)], cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == MISSING_INPUT_EXIT_CODE
    assert "Traceback" not in completed.stderr
    assert completed.stderr.strip() == f"ERROR: Missing input file {expected}."
    assert list(tmp_path.iterdir()) == []
