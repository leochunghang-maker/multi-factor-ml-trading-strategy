"""Where a workflow reads and writes its market data.

Two roots, never mixed:

- **research** — ``data/``: the checked-in snapshot used by ``make research``,
  the multi-day simulation, reports and the doc-metric checks. Only an explicit
  ``python src/data_loader.py`` (without ``--paper``) rewrites it.
- **paper** — ``data/paper/`` by default (gitignored): refreshed by the daily
  paper workflow and ``python src/data_loader.py --paper``. Override with
  ``paper_trading.data_dir`` in the platform config or the ``PAPER_DATA_DIR``
  environment variable. It may not point at the research directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESEARCH_DATA_DIR = "data"
DEFAULT_PAPER_DATA_DIR = "data/paper"
PAPER_DATA_DIR_ENV = "PAPER_DATA_DIR"


@dataclass(frozen=True)
class DataPaths:
    root: str
    kind: str = "research"

    def _file(self, name: str) -> str:
        return str(Path(self.root) / name)

    @property
    def price_data(self) -> str:
        return self._file("price_data.csv")

    @property
    def volume_data(self) -> str:
        return self._file("volume_data.csv")

    @property
    def benchmark_prices(self) -> str:
        return self._file("benchmark_prices.csv")

    @property
    def benchmark_source(self) -> str:
        return self._file("benchmark_source.json")

    @property
    def daily_returns(self) -> str:
        return self._file("daily_returns.csv")

    @property
    def momentum(self) -> str:
        return self._file("momentum_12m.csv")

    @property
    def mean_reversion(self) -> str:
        return self._file("mean_reversion_1m.csv")

    @property
    def forward_returns(self) -> str:
        return self._file("forward_returns_1m.csv")

    @property
    def ml_dataset(self) -> str:
        return self._file("ml_dataset.csv")

    @property
    def manifest(self) -> str:
        return self._file("data_manifest.json")

    @property
    def quality_report(self) -> str:
        """Research keeps its committed report under reports/; paper writes next to its data."""
        if self.kind == "research":
            return "reports/data_quality_report.md"
        return self._file("data_quality_report.md")

    def dated_inputs(self) -> dict[str, str]:
        return {
            "price_data": self.price_data,
            "volume_data": self.volume_data,
            "benchmark_prices": self.benchmark_prices,
        }

    @classmethod
    def research(cls) -> "DataPaths":
        return cls(RESEARCH_DATA_DIR, "research")

    @classmethod
    def paper(cls, root: str | Path | None = None, config: dict[str, Any] | None = None) -> "DataPaths":
        """Paper data root: explicit ``root`` > $PAPER_DATA_DIR > config > ``data/paper``."""
        if root is None:
            root = os.environ.get(PAPER_DATA_DIR_ENV)
        if root is None:
            if config is None:
                from src.operations.configuration import load_platform_config

                config = load_platform_config()
            root = (config.get("paper_trading") or {}).get("data_dir", DEFAULT_PAPER_DATA_DIR)
        root = str(root)
        if Path(root).resolve() == Path(RESEARCH_DATA_DIR).resolve():
            raise ValueError(
                f"paper data dir {root!r} is the research data dir; paper refreshes must not "
                "rewrite the checked-in research snapshot"
            )
        return cls(root, "paper")
