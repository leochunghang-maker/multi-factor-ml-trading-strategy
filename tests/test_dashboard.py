"""Dashboard text and headline numbers stay consistent with README and the code.

Renders the Streamlit app headlessly (AppTest; nothing is served or traded) and
checks the headline metrics against the doc-metric values README quotes.
"""

import re
from pathlib import Path

import pytest

from dashboard.data import RESEARCH_BOOKS, book_headline_metrics, load_strategy_returns
from src.reporting.doc_metrics import expected_metrics, num, pct

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_TEXT = [ROOT / "dashboard" / name for name in ("app.py", "charts.py", "README.md")]
PREFIX = {"multi_factor_results.csv": "mf", "walk_forward_period_returns.csv": "wf"}


@pytest.mark.parametrize("file_name", list(RESEARCH_BOOKS.values()))
def test_headline_metrics_match_readme_doc_metrics(file_name: str) -> None:
    metrics = book_headline_metrics(load_strategy_returns(file_name))
    expected = expected_metrics()
    prefix = PREFIX[file_name]
    assert pct(2)(metrics["annual_return"]) == expected[f"{prefix}.annual_return"]
    assert pct(2)(metrics["max_drawdown"]) == expected[f"{prefix}.max_drawdown"]
    assert num(2)(metrics["sharpe_ratio"]) == expected[f"{prefix}.sharpe"]
    assert pct(2)(metrics["cumulative_return"]) == expected[f"{prefix}.cumulative"]
    assert str(int(metrics["months"])) == expected[f"{prefix}.n"]
    if f"{prefix}.turnover_avg" in expected:
        assert pct(2)(metrics["average_one_way_turnover"]) == expected[f"{prefix}.turnover_avg"]


def test_dashboard_text_has_no_hardcoded_numbers_and_is_simulation_only() -> None:
    for path in DASHBOARD_TEXT:
        text = path.read_text()
        # Numbers must come from result files, never be typed into labels.
        assert not re.search(r"(?<![.\w])\d+(\.\d+)?\s?%", text), f"hard-coded percentage in {path.name}"
        assert "Institutional" not in text, f"marketing title left in {path.name}"
        # Diagnostics describe the reported book (tests/test_reported_book.py), not a stand-in.
        assert "illustrative" not in text.lower(), f"stand-in book wording left in {path.name}"
    app = (ROOT / "dashboard" / "app.py").read_text()
    for phrase in ("no broker connection", "no live-money trading", "simulated paper account"):
        assert phrase in app
    readme = " ".join((ROOT / "dashboard" / "README.md").read_text().split())
    assert "results/paper/" in readme and "does not place orders" in readme


def test_app_renders_headlessly_with_simulation_only_caption() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    assert not app.exception, app.exception
    captions = " ".join(c.value for c in app.caption)
    assert "no live-money trading" in captions
    labels = [m.label for m in app.metric]
    assert labels.count("Sharpe Ratio") == len(RESEARCH_BOOKS)
    sharpe_values = [m.value for m in app.metric if m.label == "Sharpe Ratio"]
    expected = expected_metrics()
    assert [num(2)(float(v)) for v in sharpe_values] == [expected["mf.sharpe"], expected["wf.sharpe"]]
