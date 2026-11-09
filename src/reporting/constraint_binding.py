"""How often, and by how much, the paper guardrails bind on the reported book.

Descriptive only. ``constrain_weights`` (src/portfolio/constraints.py) records
``bind_*`` diagnostics for each step on that step's own input, in application
order: liquidity filter, per-name volatility sizing, position cap, sector gross
cap, gross cap, renormalisation, turnover cap, ADV participation cap. This
module summarises them per rebalance. It changes no weights, limits or
research numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.portfolio.constraints import PortfolioConstraints

TOLERANCE = 1e-9
REPORT_TITLE = "# Paper Constraint Binding Report (reported multi-factor book)"


def _step(
    name: str,
    limit: str,
    amount: pd.Series,
    unit: str,
    detail: str,
    binding: pd.Series | None = None,
) -> dict:
    amount = amount.astype(float)
    binds = (amount > TOLERANCE) if binding is None else binding.astype(bool)
    when = amount[binds]
    return {
        "constraint": name,
        "limit": limit,
        "rebalances": int(len(amount)),
        "binding_rebalances": int(binds.sum()),
        "binding_share": float(binds.mean()) if len(amount) else np.nan,
        "mean_when_binding": float(when.mean()) if not when.empty else 0.0,
        "median_when_binding": float(when.median()) if not when.empty else 0.0,
        "max": float(amount.max()) if len(amount) else 0.0,
        "unit": unit,
        "detail": detail,
    }


def binding_summary(
    diagnostics: pd.DataFrame,
    constraints: PortfolioConstraints | None = None,
) -> pd.DataFrame:
    """One row per guardrail: rebalances where it changed the book, and by how much."""
    c = constraints or PortfolioConstraints()
    d = diagnostics
    zero = pd.Series(0.0, index=d.index)

    def col(name: str) -> pd.Series:
        return d[name] if name in d else zero

    wanted = col("bind_turnover_wanted_traded_gross")
    turnover_cut = (wanted - c.max_turnover).clip(lower=0.0).where(d["turnover_limited"].astype(bool), 0.0)

    def names(key: str, what: str = "names") -> str:
        counts = col(key)
        if not (counts > 0).any():
            return f"no {what}"
        return f"{what} per binding rebalance: mean {counts[counts > 0].mean():.1f}, max {int(counts.max())}"

    rows = [
        _step("Liquidity filter", f"63-day ADV$ ≥ ${c.min_average_daily_dollar_volume / 1e6:,.0f}M",
              col("bind_liquidity_gross_removed"), "gross removed (NAV)",
              f"{names('bind_liquidity_held_names_removed', 'held names removed')}; universe names below "
              f"the threshold on {int((col('liquidity_removed_positions') > 0).sum())} rebalances "
              f"(max {int(col('liquidity_removed_positions').max())})"),
        _step("Per-name volatility sizing", f"scale to ≤ {c.target_volatility:.0%} annualized vol",
              col("bind_vol_sizing_gross_trimmed"), "gross trimmed (NAV)", names("bind_vol_sizing_names_scaled")),
        _step("Position cap", f"\\|w\\| ≤ {c.max_position_weight:.0%}",
              col("bind_position_gross_trimmed"), "gross trimmed (NAV)", names("bind_position_names_clipped")),
        _step("Sector gross cap", f"sector gross ≤ {c.max_sector_weight:.0%}",
              col("bind_sector_gross_trimmed"), "gross trimmed (NAV)",
              f"sectors scaled per binding rebalance: max {int(col('bind_sector_count_scaled').max())}; "
              f"largest sector gross on input: mean {col('bind_sector_max_gross_before').mean():.1%}, "
              f"max {col('bind_sector_max_gross_before').max():.1%}"),
        _step("Gross cap", f"gross ≤ {c.max_gross_exposure:.0%}",
              col("bind_gross_cap_gross_trimmed"), "gross trimmed (NAV)",
              f"gross on input: mean {col('bind_gross_cap_gross_before').mean():.1%}, "
              f"max {col('bind_gross_cap_gross_before').max():.1%}"),
        _step("Renormalisation", "gross ≤ proposed gross",
              col("bind_normalize_gross_trimmed"), "gross trimmed (NAV)", "only scales down"),
        _step("Turnover cap", f"traded gross ≤ {c.max_turnover:.0%}",
              turnover_cut, "traded gross cut (NAV)",
              f"wanted traded gross: mean {wanted.mean():.1%}, max {wanted.max():.1%}",
              binding=d["turnover_limited"]),
        _step("ADV participation cap", f"\\|Δw\\| × ${c.capacity_nav:,.0f} ≤ {c.max_adv_participation:.0%} of ADV$",
              col("bind_capacity_traded_trimmed"), "traded gross cut (NAV)",
              names("capacity_clipped_positions", "names clipped")),
    ]
    return pd.DataFrame(rows)


def exposure_comparison(
    proposed: pd.DataFrame,
    constrained: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    constraints: PortfolioConstraints | None = None,
) -> pd.DataFrame:
    """Gross/net/position/sector levels before and after the guardrails, plus residual breaches."""
    from src.config import SECTOR_MAP

    c = constraints or PortfolioConstraints()
    sectors = sector_map or c.sector_map or SECTOR_MAP
    rows = []
    for label, w in (("proposed", proposed), ("paper-constrained", constrained)):
        sector_gross = w.abs().T.groupby(lambda s: sectors.get(s, "Unknown")).sum().T.max(axis=1)
        gross = w.abs().sum(axis=1)
        net = w.sum(axis=1)
        largest = w.abs().max(axis=1)
        rows.append({
            "book": label,
            "gross_mean": gross.mean(), "gross_min": gross.min(), "gross_max": gross.max(),
            "net_mean": net.mean(), "net_min": net.min(), "net_max": net.max(),
            "largest_name_mean": largest.mean(), "largest_name_max": largest.max(),
            "largest_sector_gross_mean": sector_gross.mean(), "largest_sector_gross_max": sector_gross.max(),
            "position_cap_breaches": int((largest > c.max_position_weight + TOLERANCE).sum()),
            "sector_cap_breaches": int((sector_gross > c.max_sector_weight + TOLERANCE).sum()),
            "gross_cap_breaches": int((gross > c.max_gross_exposure + TOLERANCE).sum()),
        })
    return pd.DataFrame(rows).set_index("book")


def vol_sizing_comparison(
    with_vol: tuple[pd.DataFrame, pd.DataFrame],
    without_vol: tuple[pd.DataFrame, pd.DataFrame],
    proposed: pd.DataFrame,
    constraints: PortfolioConstraints | None = None,
) -> pd.DataFrame:
    """Same guardrails with and without the per-name volatility-sizing step.

    Each argument pair is ``(constrained_weights, diagnostics)`` from
    ``constrain_weight_history`` (without = ``daily_returns=None``). Counts are
    rebalances where a cap changed the book on its input, then residual
    breaches after every step.
    """
    rows = {}
    for label, (weights, diagnostics) in (
        ("with per-name vol sizing (current)", with_vol),
        ("without per-name vol sizing", without_vol),
    ):
        steps = binding_summary(diagnostics, constraints).set_index("constraint")["binding_rebalances"]
        exposure = exposure_comparison(proposed, weights, constraints=constraints).loc["paper-constrained"]
        gross_in = diagnostics["bind_gross_cap_gross_before"]
        rows[label] = {
            "gross_into_gross_cap_mean": float(gross_in.mean()),
            "gross_mean": exposure["gross_mean"], "gross_min": exposure["gross_min"],
            "gross_max": exposure["gross_max"], "net_mean": exposure["net_mean"],
            "largest_name_mean": exposure["largest_name_mean"],
            "largest_sector_gross_mean": exposure["largest_sector_gross_mean"],
            "position_cap_binds": int(steps["Position cap"]),
            "sector_cap_binds": int(steps["Sector gross cap"]),
            "gross_cap_binds": int(steps["Gross cap"]),
            "turnover_cap_binds": int(steps["Turnover cap"]),
            "position_cap_breaches": int(exposure["position_cap_breaches"]),
            "sector_cap_breaches": int(exposure["sector_cap_breaches"]),
            "gross_cap_breaches": int(exposure["gross_cap_breaches"]),
        }
    return pd.DataFrame(rows).T


def render_markdown(
    summary: pd.DataFrame,
    exposures: pd.DataFrame,
    diagnostics: pd.DataFrame,
    vol_comparison: pd.DataFrame | None = None,
) -> str:
    start, end = diagnostics.index.min(), diagnostics.index.max()
    n = len(diagnostics)
    lines = [
        REPORT_TITLE,
        "",
        "Generated by `make report` (`python src/generate_report.py`). Descriptive only: no weights,",
        "limits or research numbers change. The reported book is the beta-neutral, vol-targeted",
        "multi-factor long/short book (`results/multi_factor_results.csv`); the guardrails are",
        "`PortfolioConstraints` defaults applied by `constrain_weight_history`, the layer behind the",
        "dashboard's \"paper-constrained\" charts. The simulated paper account itself trades a separate",
        "long-only top-10 book (`reports/multi_day_simulation_report.md`).",
        "",
        f"Rebalances: {n} month-ends, {pd.Timestamp(start).date()} → {pd.Timestamp(end).date()}.",
        "Each step is measured on its own input, in application order, so a later step only sees",
        "what earlier steps left (e.g. the gross cap acts after volatility sizing). Amounts are",
        "fractions of NAV. The first rebalance trades from cash, so it counts toward the turnover cap.",
        "",
        "## How often and by how much each guardrail binds",
        "",
        "| Step | Limit | Binding rebalances | Mean when binding | Median when binding | Max | Unit | Detail |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['constraint']} | {r['limit']} | {r['binding_rebalances']} / {r['rebalances']} "
            f"({r['binding_share']:.0%}) | {r['mean_when_binding']:.2%} | {r['median_when_binding']:.2%} | "
            f"{r['max']:.2%} | {r['unit']} | {r['detail']} |"
        )
    lines += [
        "",
        "## Exposure before and after the guardrails",
        "",
        "There is no net-exposure limit in `PortfolioConstraints`; net is shown for reference.",
        "",
        "| Book | Gross mean (min–max) | Net mean (min–max) | Largest name mean / max | Largest sector gross mean / max | Rebalances over position / sector / gross cap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for book, r in exposures.iterrows():
        lines.append(
            f"| {book} | {r['gross_mean']:.1%} ({r['gross_min']:.1%}–{r['gross_max']:.1%}) | "
            f"{r['net_mean']:.1%} ({r['net_min']:.1%}–{r['net_max']:.1%}) | "
            f"{r['largest_name_mean']:.1%} / {r['largest_name_max']:.1%} | "
            f"{r['largest_sector_gross_mean']:.1%} / {r['largest_sector_gross_max']:.1%} | "
            f"{int(r['position_cap_breaches'])} / {int(r['sector_cap_breaches'])} / {int(r['gross_cap_breaches'])} |"
        )
    residual = exposures.loc["paper-constrained", ["position_cap_breaches", "sector_cap_breaches", "gross_cap_breaches"]]
    lines.append("")
    if residual.sum() > 0:
        lines += [
            "Residual breaches in the paper-constrained row come from the turnover blend: when the cap",
            "binds, the book moves only part of the way from the previous weights, which can leave a",
            "name or sector above its cap until a later rebalance.",
        ]
    else:
        lines += [
            "The paper-constrained book ends every rebalance within the position, sector gross and",
            "gross caps, including after the turnover blend.",
        ]
    lines.append("")
    if vol_comparison is not None:
        lines += [
            "## Effect of the per-name volatility-sizing step",
            "",
            "The same guardrails run twice on the same proposed book: as configured, and with the",
            "per-name volatility-sizing step skipped (counterfactual for description only; nothing",
            "uses it). The proposed book is already volatility-targeted at the portfolio level.",
            "",
            "| Guardrails | Gross into gross cap (mean) | Final gross mean (min–max) | Final net mean | Largest name / sector gross (mean) | Rebalances where position / sector / gross / turnover cap binds | Residual position / sector / gross breaches |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, r in vol_comparison.iterrows():
            lines.append(
                f"| {label} | {r['gross_into_gross_cap_mean']:.1%} | "
                f"{r['gross_mean']:.1%} ({r['gross_min']:.1%}–{r['gross_max']:.1%}) | {r['net_mean']:.1%} | "
                f"{r['largest_name_mean']:.1%} / {r['largest_sector_gross_mean']:.1%} | "
                f"{int(r['position_cap_binds'])} / {int(r['sector_cap_binds'])} / {int(r['gross_cap_binds'])} / "
                f"{int(r['turnover_cap_binds'])} | "
                f"{int(r['position_cap_breaches'])} / {int(r['sector_cap_breaches'])} / {int(r['gross_cap_breaches'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def save_binding_report(
    diagnostics: pd.DataFrame,
    proposed: pd.DataFrame,
    constrained: pd.DataFrame,
    csv_path: str | Path,
    markdown_path: str | Path,
    without_vol_sizing: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    summary = binding_summary(diagnostics)
    exposures = exposure_comparison(proposed, constrained)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    comparison = None
    if without_vol_sizing is not None:
        comparison = vol_sizing_comparison((constrained, diagnostics), without_vol_sizing, proposed)
        comparison.to_csv(Path(csv_path).with_name("constraint_binding_vol_sizing.csv"))
    summary.to_csv(csv_path, index=False)
    Path(markdown_path).write_text(render_markdown(summary, exposures, diagnostics, comparison))
    return summary
