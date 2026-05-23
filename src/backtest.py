import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtests.factor_backtests import run_momentum_backtest
from src.config import MOMENTUM_BACKTEST_PATH, MOMENTUM_EQUITY_CURVE_PATH
from src.reporting.charts import save_equity_curve


def main() -> None:
    results, curve, metrics = run_momentum_backtest()

    results.to_csv(MOMENTUM_BACKTEST_PATH)

    save_equity_curve(
        curve,
        MOMENTUM_EQUITY_CURVE_PATH,
        "Momentum Strategy Equity Curve",
    )

    print("Monthly Momentum Long-Short Backtest Completed")
    print()
    print("Annual Return:", metrics["annual_return"])
    print("Annual Volatility:", metrics["annual_volatility"])
    print("Sharpe Ratio:", metrics["sharpe_ratio"])
    print("Max Drawdown:", metrics["max_drawdown"])
    print()
    print("Number of Monthly Trades:", len(results))
    print()
    print(f"Equity curve saved to {MOMENTUM_EQUITY_CURVE_PATH}")


if __name__ == "__main__":
    main()
