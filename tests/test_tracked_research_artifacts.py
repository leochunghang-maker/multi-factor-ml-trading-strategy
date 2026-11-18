"""Committed research artifacts under data/ and results/ must be visible to git.

The research snapshot and the result files that back README / report numbers are
checked in. Broad ignore rules for `data/*.csv` and the `results/` directory used
to hide them from `git add` unless someone remembered `git add -f`, so a
regenerated or newly added artifact could be left out of a commit. `.gitignore`
must un-ignore every tracked file in those trees. Local paper, simulation and
operations outputs stay ignored.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Operator-local outputs that must remain untracked.
MUST_STAY_IGNORED = [
    "data/paper/price_data.csv",
    "data/paper/data_manifest.json",
    "results/live/latest_signals.csv",
    "results/live/latest_model.pkl",
    "results/operations/local_system_status.md",
    "results/operations/local_data_quality_report.md",
    "results/paper/journal.md",
    "results/paper/status.md",
    "results/simulation/broker_state.json",
    "results/simulation/multi_day/equity.csv",
    "results/reports/summary_report.csv",
    "results/risk_managed_equity_curve.png",
]

# Committed outputs of `make research` (and the frozen snapshot it reads).
# Local-only steps such as risk_managed_equity_curve.png are intentionally absent.
MUST_BE_TRACKED = [
    "data/price_data.csv",
    "data/volume_data.csv",
    "data/benchmark_prices.csv",
    "data/benchmark_source.json",
    "data/data_manifest.json",
    "data/daily_returns.csv",
    "data/momentum_12m.csv",
    "data/mean_reversion_1m.csv",
    "data/forward_returns_1m.csv",
    "data/ml_dataset.csv",
    "results/momentum_ic_results.csv",
    "results/momentum_backtest.csv",
    "results/equity_curve.png",
    "results/ml_predictions.csv",
    "results/ml_strategy_equity_curve.png",
    "results/multi_factor_results.csv",
    "results/multi_factor_equity_curve.csv",
    "results/strategy_vs_benchmark.png",
    "results/walk_forward_period_returns.csv",
    "results/walk_forward_metrics.csv",
    "results/walk_forward_ml_strategy.png",
    "results/reports/nested_model_selection.csv",
    "results/significance_report.csv",
    "results/benchmark_relative_report.csv",
    "results/multi_factor_beta_diagnosis.csv",
    "reports/data_quality_report.md",
    "reports/multi_day_simulation_report.md",
    "reports/system_status.md",
]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _tracked(*paths: str) -> list[str]:
    completed = _git("ls-files", "-z", *paths)
    assert completed.returncode == 0, completed.stderr
    return [path for path in completed.stdout.split("\0") if path]


def test_research_pipeline_artifacts_are_tracked() -> None:
    tracked = set(_tracked("data/", "results/", "reports/"))
    missing = [path for path in MUST_BE_TRACKED if path not in tracked]
    assert not missing, (
        "make research artifacts are not tracked; add them and a .gitignore "
        f"exception if needed: {missing}"
    )


def test_tracked_data_and_results_are_not_ignored_by_gitignore() -> None:
    tracked = _tracked("data/", "results/")
    assert tracked, "expected committed research files under data/ and results/"
    hidden = []
    for path in tracked:
        # Without -v, exit 0 means the path is ignored; a later ! exception
        # yields exit 1 (same as a path that never matched).
        completed = _git("check-ignore", "--no-index", path)
        if completed.returncode == 0:
            detail = _git("check-ignore", "--no-index", "-v", path).stdout.strip()
            hidden.append(detail or path)
    assert not hidden, (
        "committed research artifacts are hidden by .gitignore; add a negation "
        "rule so `git add` works without -f:\n  " + "\n  ".join(hidden)
    )


def test_local_paper_and_sim_outputs_stay_gitignored() -> None:
    visible = [
        path
        for path in MUST_STAY_IGNORED
        if _git("check-ignore", "--no-index", path).returncode != 0
    ]
    assert not visible, f"these local outputs must stay gitignored: {visible}"
