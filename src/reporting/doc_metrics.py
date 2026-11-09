"""Keep numbers quoted in README.md / reports/*.md in sync with the result files.

Docs mark a quoted number with an HTML comment pair (invisible on GitHub)::

    Sharpe <!--m:mf.sharpe-->0.25<!--/m-->

``expected_metrics()`` recomputes every key from committed artifacts only
(results/*.csv, research/trial_log.csv, reports/multi_day_simulation_report.md),
formatted exactly as the docs print it. ``check_documents`` reports unknown keys,
mismatched values and required keys that are not quoted anywhere; ``fix_documents``
rewrites mismatched values in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

MARKER = re.compile(r"<!--m:([A-Za-z0-9_.\-]+)-->(.*?)<!--/m-->")
DEFAULT_DOCUMENTS = ("README.md", "reports/quant_research_report.md")

STRATEGIES = {"walk_forward_ml": "wf", "multi_factor": "mf", "multi_day_sim": "sim"}
BENCHMARKS = {"equal_weight_universe": "ew", "SPY": "spy"}

# Keys that must be quoted (and therefore checked) at least once in README.md.
REQUIRED_IN_README = (
    "mf.annual_return", "mf.annual_volatility", "mf.sharpe", "mf.max_drawdown", "mf.cumulative",
    "mf.ending_wealth", "wf.annual_return", "wf.annual_volatility", "wf.sharpe", "wf.max_drawdown",
    "wf.cumulative", "sig.wf.dsr", "sig.mf.dsr", "sig.sim.dsr_pooled", "trials.n",
    "rel.wf.spy.beta", "rel.mf.spy.beta", "rel.mf.spy.alpha_t", "sim.final_nav", "sim.sharpe",
)


def _clean(text: str) -> str:
    return "0.00" if text in {"-0.00", "-0.0"} else text.replace("-0.00%", "0.00%")


def pct(digits: int) -> Callable[[float], str]:
    return lambda x: _clean(f"{x:.{digits}%}")


def num(digits: int) -> Callable[[float], str]:
    return lambda x: _clean(f"{x:.{digits}f}")


def ci(digits: int) -> Callable[[tuple[float, float]], str]:
    return lambda pair: f"[{num(digits)(pair[0])}, {num(digits)(pair[1])}]"


def _metrics(returns: pd.Series, periods: int = 12) -> dict[str, float]:
    from src.reporting.metrics import calculate_performance_metrics

    m = calculate_performance_metrics(returns)
    return {
        "annual_return": float(m["annual_return"]),
        "annual_volatility": float(m["annual_volatility"]),
        "sharpe": float(m["sharpe_ratio"]),
        "max_drawdown": float(m["max_drawdown"]),
        "cumulative": float((1.0 + returns).prod() - 1.0),
    }


def _book_values(prefix: str, frame: pd.DataFrame) -> dict[str, str]:
    r = frame["portfolio_return"].astype(float)
    m = _metrics(r)
    out = {
        f"{prefix}.annual_return": pct(2)(m["annual_return"]),
        f"{prefix}.annual_volatility": pct(2)(m["annual_volatility"]),
        f"{prefix}.sharpe": num(2)(m["sharpe"]),
        f"{prefix}.max_drawdown": pct(2)(m["max_drawdown"]),
        f"{prefix}.cumulative": pct(2)(m["cumulative"]),
        f"{prefix}.ending_wealth": num(4)(1.0 + m["cumulative"]),
        f"{prefix}.n": str(len(r)),
        f"{prefix}.start": str(pd.Timestamp(frame.index.min()).date()),
        f"{prefix}.end": str(pd.Timestamp(frame.index.max()).date()),
    }
    if "vol_scale" in frame:
        out[f"{prefix}.vol_scale"] = num(2)(frame["vol_scale"].mean())
    if "traded_gross_exposure" in frame and len(frame) > 1:
        traded = frame["traded_gross_exposure"].astype(float)
        out[f"{prefix}.turnover_avg"] = pct(2)(traded.iloc[1:].mean() / 2.0)
        out[f"{prefix}.turnover_latest"] = pct(2)(traded.iloc[-1] / 2.0)
    if "ex_ante_beta" in frame:
        out[f"{prefix}.ex_ante_neutral_count"] = str(int((frame["ex_ante_beta"].abs() <= 0.01).sum()))
    return out


def _significance_values(sig: pd.DataFrame, n_trials: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, short in {**STRATEGIES, "reference_equal_weight_universe": "ref"}.items():
        rows = sig[sig["strategy"] == name]
        if rows.empty:
            continue
        current = rows[rows["n_trials"] == n_trials]
        row = (current if not current.empty else rows).iloc[0]
        out.update({
            f"sig.{short}.n_obs": str(int(row["n_obs"])),
            f"sig.{short}.sharpe": num(2)(row["sharpe_annualized"]),
            f"sig.{short}.ci": ci(2)((row["ci_lower"], row["ci_upper"])),
            f"sig.{short}.p_le_zero": pct(1)(row["prob_sharpe_le_zero"]),
            f"sig.{short}.psr": num(2)(row["psr_vs_zero"]),
        })
        if short == "ref":
            continue
        out.update({
            f"sig.{short}.sr0": num(2)(row["sr0_annualized"]),
            f"sig.{short}.dsr": num(2)(row["deflated_sharpe_ratio"]),
            f"sig.{short}.dsr_pooled": num(2)(row["deflated_sharpe_ratio_pooled_var"]),
        })
        n200 = rows[rows["n_trials"] == 200]
        if not n200.empty:
            out[f"sig.{short}.dsr_n200"] = num(2)(n200.iloc[0]["deflated_sharpe_ratio"])
            out[f"sig.{short}.dsr_pooled_n200"] = num(2)(n200.iloc[0]["deflated_sharpe_ratio_pooled_var"])
    return out


def _relative_values(rel: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, row in rel.iterrows():
        s = STRATEGIES.get(row["strategy"])
        b = BENCHMARKS.get(row["benchmark"])
        if s is None or b is None:
            continue
        p = f"rel.{s}.{b}"
        out.update({
            f"{p}.alpha": pct(1)(row["alpha_annualized"]),
            f"{p}.alpha_t": num(2)(row["alpha_t_hac"]),
            f"{p}.beta": num(2)(row["beta"]),
            f"{p}.ir": num(2)(row["information_ratio"]),
            f"{p}.ir_ci": ci(2)((row["ir_ci_lower"], row["ir_ci_upper"])),
            f"{p}.ir_dsr": num(2)(row["ir_deflated"]),
            f"{p}.hedged_is": num(2)(row["hedged_sharpe_in_sample"]),
            f"{p}.hedged_causal": num(2)(row["hedged_sharpe_causal"]),
            f"{p}.hedged_causal_dsr": num(2)(row["hedged_causal_deflated"]),
        })
    return out


def _trial_values(root: Path) -> tuple[dict[str, str], int]:
    from src.config import SIGNIFICANCE_TRIAL_LOG_PATH, SIGNIFICANCE_TRIAL_PADDING
    from src.reporting.trial_log import count_trials, family_sharpes, load_trial_log, pooled_sharpes

    log = load_trial_log(root / SIGNIFICANCE_TRIAL_LOG_PATH)
    n = count_trials(log, padding=SIGNIFICANCE_TRIAL_PADDING)
    pooled = pooled_sharpes(log)
    out = {
        "trials.n": str(n),
        "trials.logged": str(int(log["trial_increment"].sum())),
        "trials.padding": str(int(SIGNIFICANCE_TRIAL_PADDING)),
        "trials.rows": str(len(log)),
        "trials.pooled_count": str(len(pooled)),
        "trials.pooled_sd": num(2)(np.std(pooled, ddof=1)),
    }
    for family, short in STRATEGIES.items():
        values = family_sharpes(log, family)
        out[f"trials.{short}.count"] = str(int((log["family"] == family).sum()))
        if len(values) > 1:
            out[f"trials.{short}.sd"] = num(2)(np.std(values, ddof=1))
    return out, n


def _sim_values(root: Path) -> dict[str, str]:
    path = root / "reports" / "multi_day_simulation_report.md"
    if not path.exists():
        return {}
    text = path.read_text()

    def grab(label: str) -> str | None:
        match = re.search(rf"^- {re.escape(label)}[^:]*: (.+)$", text, flags=re.M)
        return match.group(1).strip() if match else None

    out = {}
    nav = grab("Portfolio value")
    if nav is not None:
        out["sim.final_nav"] = f"{float(nav):,.2f}"
    for key, label in {
        "sim.cagr": "CAGR",
        "sim.sharpe": "Sharpe",
        "sim.max_drawdown": "Max drawdown",
        "sim.days": "Total simulated days",
        "sim.trades": "Total trades",
    }.items():
        value = grab(label)
        if value is not None:
            out[key] = value
    halts = re.search(r"^- DRAWDOWN_HALT: (\d+)$", text, flags=re.M)
    if halts:
        out["sim.drawdown_halts"] = halts.group(1)
    return out


def expected_metrics(root: str | Path = ".") -> dict[str, str]:
    """Every checkable key → the string the docs should show."""
    from src.config import (
        BENCHMARK_RELATIVE_REPORT_PATH,
        MULTI_FACTOR_RESULTS_PATH,
        SIGNIFICANCE_REPORT_PATH,
        WALK_FORWARD_PERIOD_RETURNS_PATH,
    )

    root = Path(root)
    values: dict[str, str] = {}
    mf = pd.read_csv(root / MULTI_FACTOR_RESULTS_PATH, parse_dates=["date"]).set_index("date")
    wf = pd.read_csv(root / WALK_FORWARD_PERIOD_RETURNS_PATH, parse_dates=["date"]).set_index("date")
    values.update(_book_values("mf", mf))
    values.update(_book_values("wf", wf))
    trials, n = _trial_values(root)
    values.update(trials)
    values.update(_significance_values(pd.read_csv(root / SIGNIFICANCE_REPORT_PATH), n))
    values.update(_relative_values(pd.read_csv(root / BENCHMARK_RELATIVE_REPORT_PATH)))
    values.update(_sim_values(root))
    return values


@dataclass
class DocIssue:
    path: str
    line: int
    key: str
    found: str
    expected: str | None
    kind: str  # mismatch | unknown_key | missing_required

    def __str__(self) -> str:
        if self.kind == "unknown_key":
            return f"{self.path}:{self.line}: unknown metric key {self.key!r}"
        if self.kind == "missing_required":
            return f"{self.path}: required metric {self.key!r} is not quoted with a marker"
        return f"{self.path}:{self.line}: {self.key} shows {self.found!r}, results say {self.expected!r}"


def scan(text: str):
    """Yield (line_number, key, value, match) for every marker in ``text``."""
    for match in MARKER.finditer(text):
        yield text.count("\n", 0, match.start()) + 1, match.group(1), match.group(2), match


def check_documents(
    documents=DEFAULT_DOCUMENTS,
    root: str | Path = ".",
    expected: dict[str, str] | None = None,
) -> list[DocIssue]:
    root = Path(root)
    expected = expected_metrics(root) if expected is None else expected
    issues: list[DocIssue] = []
    for doc in documents:
        text = (root / doc).read_text()
        seen = set()
        for line, key, value, _ in scan(text):
            seen.add(key)
            if key not in expected:
                issues.append(DocIssue(doc, line, key, value, None, "unknown_key"))
            elif value != expected[key]:
                issues.append(DocIssue(doc, line, key, value, expected[key], "mismatch"))
        if doc == "README.md":
            for key in REQUIRED_IN_README:
                if key not in seen:
                    issues.append(DocIssue(doc, 0, key, "", expected.get(key), "missing_required"))
    return issues


def fix_documents(
    documents=DEFAULT_DOCUMENTS,
    root: str | Path = ".",
    expected: dict[str, str] | None = None,
) -> int:
    """Rewrite mismatched marker values; returns the number of values changed."""
    root = Path(root)
    expected = expected_metrics(root) if expected is None else expected
    changed = 0
    for doc in documents:
        path = root / doc
        text = path.read_text()

        def repl(match: re.Match) -> str:
            nonlocal changed
            key, value = match.group(1), match.group(2)
            if key in expected and value != expected[key]:
                changed += 1
                return f"<!--m:{key}-->{expected[key]}<!--/m-->"
            return match.group(0)

        new = MARKER.sub(repl, text)
        if new != text:
            path.write_text(new)
    return changed
