import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    MULTI_FACTOR_RESULTS_PATH,
    RISK_MANAGED_EQUITY_CURVE_PATH,
    TARGET_VOLATILITY,
    VOLATILITY_LOOKBACK,
)
from src.data.io import load_return_series
from src.reporting.charts import save_equity_curve
from src.reporting.metrics import calculate_performance_metrics, equity_curve
from src.risk import apply_volatility_target


def main() -> None:
    returns = load_return_series(
        MULTI_FACTOR_RESULTS_PATH,
        "portfolio_return",
    )

    risk_managed_returns = apply_volatility_target(
        returns,
        target_volatility=TARGET_VOLATILITY,
        lookback=VOLATILITY_LOOKBACK,
    )

    curve = equity_curve(risk_managed_returns)
    metrics = calculate_performance_metrics(risk_managed_returns)

    print("Risk-Managed Strategy Completed")
    print()
    print("Annual Return:", metrics["annual_return"])
    print("Annual Volatility:", metrics["annual_volatility"])
    print("Sharpe Ratio:", metrics["sharpe_ratio"])
    print("Max Drawdown:", metrics["max_drawdown"])

    save_equity_curve(
        curve,
        RISK_MANAGED_EQUITY_CURVE_PATH,
        "Risk-Managed Strategy Equity Curve",
    )

    print()
    print(f"Chart saved to {RISK_MANAGED_EQUITY_CURVE_PATH}")


if __name__ == "__main__":
    main()
