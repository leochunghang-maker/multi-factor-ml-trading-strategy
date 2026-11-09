import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.multi_factor_backtest import main
from src.data.required import run_cli


if __name__ == "__main__":
    run_cli(main)
