import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtests.ml_backtests import run_prediction_backtest
from src.config import ML_EQUITY_CURVE_PATH
from src.reporting.charts import save_equity_curve


def main() -> None:
    _, curve, metrics = run_prediction_backtest()

    print("ML Strategy Backtest Completed")
    print()
    print("Annual Return:", metrics["annual_return"])
    print("Annual Volatility:", metrics["annual_volatility"])
    print("Sharpe Ratio:", metrics["sharpe_ratio"])
    print("Max Drawdown:", metrics["max_drawdown"])

    save_equity_curve(
        curve,
        ML_EQUITY_CURVE_PATH,
        "ML Strategy Equity Curve",
    )

    print()
    print(f"Equity curve saved to {ML_EQUITY_CURVE_PATH}")


if __name__ == "__main__":
    main()
