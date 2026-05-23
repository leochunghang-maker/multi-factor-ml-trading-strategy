import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtests.benchmark import load_benchmark_equity_curve
from src.backtests.ml_backtests import run_walk_forward_ml_backtest
from src.config import BENCHMARK_SYMBOL, TRANSACTION_COST, WALK_FORWARD_CHART_PATH
from src.reporting.charts import save_equity_curve, save_strategy_vs_benchmark


def main() -> None:
    results, curve, metrics = run_walk_forward_ml_backtest(
        transaction_cost=TRANSACTION_COST,
    )

    print()
    print("Walk-Forward ML Backtest Completed")
    print()
    print("Annual Return:", metrics["annual_return"])
    print("Annual Volatility:", metrics["annual_volatility"])
    print("Sharpe Ratio:", metrics["sharpe_ratio"])
    print("Max Drawdown:", metrics["max_drawdown"])

    print()
    try:
        benchmark_curve = load_benchmark_equity_curve(
            BENCHMARK_SYMBOL,
            start=results.index.min(),
            end=results.index.max(),
        )
        save_strategy_vs_benchmark(
            curve,
            benchmark_curve,
            WALK_FORWARD_CHART_PATH,
            "Walk-Forward ML Strategy vs SPY",
            strategy_label="Walk-Forward ML Strategy",
            ylabel="Portfolio Value",
        )
        print(f"Chart saved to {WALK_FORWARD_CHART_PATH}")
    except Exception as exc:
        print(f"Benchmark comparison chart unavailable: {exc}")
        save_equity_curve(
            curve,
            WALK_FORWARD_CHART_PATH,
            "Walk-Forward ML Strategy",
        )
        print(f"Strategy-only chart saved to {WALK_FORWARD_CHART_PATH}")


if __name__ == "__main__":
    main()
