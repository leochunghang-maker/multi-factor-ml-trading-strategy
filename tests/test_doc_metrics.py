"""Numbers quoted in README.md / reports are marked and must match the result files."""

from __future__ import annotations

import pandas as pd
import pytest

from src.reporting.doc_metrics import (
    DEFAULT_DOCUMENTS,
    MARKER,
    REQUIRED_IN_README,
    check_documents,
    ci,
    expected_metrics,
    fix_documents,
    num,
    pct,
)


def test_formatters() -> None:
    assert pct(2)(-0.41347) == "-41.35%"
    assert pct(1)(0.0787) == "7.9%"
    assert num(2)(-0.0001) == "0.00"
    assert num(4)(1.25287) == "1.2529"
    assert ci(2)((-0.4058, 1.2736)) == "[-0.41, 1.27]"


def test_check_and_fix_round_trip(tmp_path) -> None:
    doc = tmp_path / "README.md"
    doc.write_text(
        "Sharpe **<!--m:mf.sharpe-->0.31<!--/m-->** and cum <!--m:mf.cumulative-->25.29%<!--/m-->\n"
        "bad <!--m:nope-->1<!--/m-->\n"
    )
    expected = {"mf.sharpe": "0.25", "mf.cumulative": "25.29%"}
    issues = check_documents(["README.md"], root=tmp_path, expected=expected)
    kinds = sorted(i.kind for i in issues)
    assert kinds.count("mismatch") == 1 and kinds.count("unknown_key") == 1
    assert kinds.count("missing_required") == len(REQUIRED_IN_README) - 2
    mismatch = next(i for i in issues if i.kind == "mismatch")
    assert (mismatch.line, mismatch.found, mismatch.expected) == (1, "0.31", "0.25")
    assert fix_documents(["README.md"], root=tmp_path, expected=expected) == 1
    assert "<!--m:mf.sharpe-->0.25<!--/m-->" in doc.read_text()


def test_expected_metrics_cover_required_keys() -> None:
    expected = expected_metrics()
    missing = [k for k in REQUIRED_IN_README if k not in expected]
    assert not missing


def test_walk_forward_metrics_csv_matches_period_returns() -> None:
    from src.reporting.metrics import calculate_performance_metrics

    returns = pd.read_csv("results/walk_forward_period_returns.csv")["portfolio_return"]
    computed = calculate_performance_metrics(returns)
    stored = pd.read_csv("results/walk_forward_metrics.csv").iloc[0]
    for key in ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown"]:
        assert stored[key] == pytest.approx(computed[key], rel=1e-9)


def test_committed_docs_match_committed_results() -> None:
    issues = check_documents()
    assert not issues, "\n".join(str(i) for i in issues) + "\nRun: python src/check_doc_metrics.py --fix"
    total = sum(len(MARKER.findall(open(d).read())) for d in DEFAULT_DOCUMENTS)
    assert total >= 200  # guard against markers being stripped wholesale
