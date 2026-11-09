"""Regenerate every committed research artifact in dependency order (`make research`).

Uses the checked-in data only (no download, no trading). Steps run as separate
processes so each script behaves exactly as when run by hand; the pipeline stops
at the first failure. The last step verifies that numbers quoted in README.md and
reports/*.md still match the regenerated result files.

    python scripts/run_research_pipeline.py            # full run
    python scripts/run_research_pipeline.py --list     # show the steps
    python scripts/run_research_pipeline.py --from significance
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (name, command, artifacts it (re)writes) — order matters: each step reads only
# files written by the data snapshot or by earlier steps.
STEPS: list[tuple[str, list[str], str]] = [
    ("data-check", ["src/check_data.py", "--write-report"],
     "reports/data_quality_report.md (fails on data ERRORs)"),
    ("factors", ["src/factors.py"],
     "data/daily_returns.csv, momentum_12m.csv, mean_reversion_1m.csv, forward_returns_1m.csv"),
    ("ml-dataset", ["src/ml_dataset.py"], "data/ml_dataset.csv"),
    ("momentum-ic", ["src/research_test.py"], "results/momentum_ic_results.csv"),
    ("momentum", ["src/backtest.py"], "results/momentum_backtest.csv, results/equity_curve.png"),
    ("ml-baseline", ["src/train_ml_model.py"], "results/ml_predictions.csv"),
    ("ml-backtest", ["src/ml_backtest.py"], "results/ml_strategy_equity_curve.png"),
    ("multi-factor", ["src/multi_factor_backtest.py"],
     "results/multi_factor_results.csv, multi_factor_equity_curve.csv, strategy_vs_benchmark.png"),
    ("risk-managed", ["src/risk_managed_backtest.py"], "results/risk_managed_equity_curve.png (local)"),
    ("walk-forward", ["src/walk_forward_ml_backtest.py"],
     "results/walk_forward_period_returns.csv, walk_forward_metrics.csv, walk_forward_ml_strategy.png, "
     "results/reports/nested_model_selection.csv"),
    ("multi-day-sim", ["src/simulation/run_multi_day_simulation.py"],
     "results/simulation/multi_day/* (local), reports/multi_day_simulation_report.md, reports/system_status.md"),
    ("significance", ["src/run_significance_report.py", "--cached-walk-forward"],
     "results/significance_report.csv, results/benchmark_relative_report.csv"),
    ("beta-diagnosis", ["research/diagnose_multi_factor_beta.py"], "results/multi_factor_beta_diagnosis.csv"),
    ("doc-metrics", ["src/check_doc_metrics.py"], "verifies quoted numbers in README.md and reports/*.md"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Print the steps and exit.")
    parser.add_argument("--from", dest="start", default=None, help="Start at this step name.")
    args = parser.parse_args(argv)

    names = [name for name, _, _ in STEPS]
    if args.list:
        for i, (name, cmd, outputs) in enumerate(STEPS, 1):
            print(f"{i:2d}. {name:15s} python {' '.join(cmd)}\n      -> {outputs}")
        return 0
    start = names.index(args.start) if args.start else 0

    total = time.time()
    for name, cmd, _ in STEPS[start:]:
        print(f"\n=== [{name}] python {' '.join(cmd)}", flush=True)
        began = time.time()
        completed = subprocess.run([sys.executable, *cmd], cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            print(f"=== [{name}] FAILED (exit {completed.returncode}); stopping.", flush=True)
            return completed.returncode
        print(f"=== [{name}] ok ({time.time() - began:.0f}s)", flush=True)
    print(f"\nResearch pipeline finished in {time.time() - total:.0f}s. Review `git diff --stat`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
