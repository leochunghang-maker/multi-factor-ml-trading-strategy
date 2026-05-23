import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtests.benchmark import load_benchmark_equity_curve
from src.backtests.factor_backtests import run_multi_factor_backtest
from src.config import (
    BENCHMARK_SYMBOL,
    MULTI_FACTOR_EQUITY_CURVE_PATH,
    MULTI_FACTOR_RESULTS_PATH,
    STRATEGY_VS_BENCHMARK_PATH,
    TRANSACTION_COST,
)
from src.reporting.charts import save_equity_curve, save_strategy_vs_benchmark


def main() -> None:
    results, curve, metrics = run_multi_factor_backtest(
        transaction_cost=TRANSACTION_COST,
    )

    print("Multi-Factor Backtest Completed")
    print()
    print("Annual Return:", metrics["annual_return"])
    print("Annual Volatility:", metrics["annual_volatility"])
    print("Sharpe Ratio:", metrics["sharpe_ratio"])
    print("Max Drawdown:", metrics["max_drawdown"])
    print("Transaction Cost Per Trade:", TRANSACTION_COST)

    try:
        benchmark_curve = load_benchmark_equity_curve(
            BENCHMARK_SYMBOL,
            start=results.index.min(),
            end=results.index.max(),
        )
        save_strategy_vs_benchmark(
            curve,
            benchmark_curve,
            STRATEGY_VS_BENCHMARK_PATH,
            "Strategy vs SPY Benchmark",
        )
        print(f"Benchmark chart saved to {STRATEGY_VS_BENCHMARK_PATH}")
    except Exception as exc:
        print(f"Benchmark chart unavailable: {exc}")
        save_equity_curve(
            curve,
            STRATEGY_VS_BENCHMARK_PATH,
            "Multi-Factor Strategy",
        )
        print(f"Strategy-only chart saved to {STRATEGY_VS_BENCHMARK_PATH}")

    results.to_csv(MULTI_FACTOR_RESULTS_PATH)
    curve.to_csv(MULTI_FACTOR_EQUITY_CURVE_PATH)


if __name__ == "__main__":
    main()
