import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import FORWARD_RETURNS_PATH, MOMENTUM_IC_RESULTS_PATH, MOMENTUM_PATH, RESEARCH_MIN_ASSETS
from src.data.io import load_factor_frame
from src.reporting.metrics import information_coefficient


def run_momentum_ic_test() -> pd.DataFrame:
    momentum = load_factor_frame(MOMENTUM_PATH)
    forward_returns = load_factor_frame(FORWARD_RETURNS_PATH)
    ic_results = []

    for date in momentum.index:
        x = momentum.loc[date]
        y = forward_returns.loc[date]
        valid = x.notna() & y.notna()

        if valid.sum() >= RESEARCH_MIN_ASSETS:
            ic, p_value = information_coefficient(x[valid], y[valid])
            ic_results.append({
                "date": date,
                "information_coefficient": ic,
                "p_value": p_value,
            })

    return pd.DataFrame(ic_results)


def main() -> None:
    ic_df = run_momentum_ic_test()
    ic_df.to_csv(MOMENTUM_IC_RESULTS_PATH, index=False)

    print("Momentum IC test completed.")
    print()
    print(ic_df.describe())
    print()
    print("Average IC:", ic_df["information_coefficient"].mean())
    print("Positive IC Ratio:", (ic_df["information_coefficient"] > 0).mean())


if __name__ == "__main__":
    main()
