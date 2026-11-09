"""One-SE fallback, equal-weight ensemble, and name-level HTB borrow in nested inner scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.nested import (
    StandardizedEnsemble,
    collect_inner_fold_diagnostics,
    fit_estimator,
    net_long_short_period_return,
    one_se_choice,
    paired_difference_tstat,
    select_model_nested,
)

FEATURES = ["momentum_12m", "short_term_momentum", "volatility_1m"]
RIDGE = {"name": "ridge_1", "kind": "ridge", "params": {"alpha": 1.0}}
RF = {
    "name": "rf_default",
    "kind": "random_forest",
    "params": {"n_estimators": 10, "max_depth": 3, "random_state": 0, "n_jobs": 1},
}


def _panel(start: str, end: str, n_tickers: int = 8, seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(seed)
    rows = []
    for date in dates:
        for i in range(n_tickers):
            feature = float(i) + rng.normal(0, 0.5)
            rows.append({
                "date": date,
                "ticker": f"T{i}",
                "momentum_12m": feature,
                "short_term_momentum": feature * 0.5,
                "volatility_1m": 0.02 + 0.001 * i,
                "target_return_1m": 0.01 * feature + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


def _rows(**scores):
    return [{"name": k, "inner_score": v} for k, v in scores.items()]


def _dated(values):
    idx = pd.date_range("2020-01-31", periods=len(values), freq="ME")
    return dict(zip(idx, values))


# ---- one-SE fallback ---------------------------------------------------------

def test_paired_difference_tstat_matches_manual() -> None:
    a = _dated([0.03, 0.01, 0.02, 0.04])
    b = _dated([0.01, 0.01, 0.01, 0.01])
    mean, se, n = paired_difference_tstat(a, b)
    diffs = np.array([0.02, 0.0, 0.01, 0.03])
    assert n == 4
    assert mean == pytest.approx(diffs.mean())
    assert se == pytest.approx(diffs.std(ddof=1) / 2.0)


def test_one_se_falls_back_when_gap_is_noise() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.05, 60)
    per_date = {
        "rf_default": _dated(base),
        "ridge_1": _dated(base + rng.normal(0.001, 0.05, 60)),  # tiny, noisy edge
    }
    name, mode, stats = one_se_choice(
        _rows(rf_default=0.1, ridge_1=0.2), per_date, default_name="rf_default", se_threshold=1.0
    )
    assert name == "rf_default"
    assert mode == "one_se_fallback_to_default"
    assert stats["best_challenger"] == "ridge_1"
    assert stats["paired_mean_diff"] <= stats["paired_se"]


def test_one_se_switches_when_challenger_clearly_better() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.05, 60)
    per_date = {
        "rf_default": _dated(base),
        "ridge_1": _dated(base + 0.02 + rng.normal(0, 0.005, 60)),
    }
    name, mode, _ = one_se_choice(
        _rows(rf_default=0.1, ridge_1=1.5), per_date, default_name="rf_default", se_threshold=1.0
    )
    assert (name, mode) == ("ridge_1", "one_se_challenger_significant")


def test_one_se_keeps_default_when_default_is_best_or_no_scores() -> None:
    assert one_se_choice(
        _rows(rf_default=0.5, ridge_1=0.2), {}, default_name="rf_default", se_threshold=1.0
    )[:2] == ("rf_default", "one_se_default_is_best")
    assert one_se_choice(
        _rows(rf_default=float("nan"), ridge_1=float("nan")), {},
        default_name="rf_default", se_threshold=1.0,
    )[:2] == ("rf_default", "fallback_no_inner_scores")


def test_higher_threshold_is_more_conservative() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.05, 60)
    per_date = {"rf_default": _dated(base), "ridge_1": _dated(base + 0.004 + rng.normal(0, 0.01, 60))}
    rows = _rows(rf_default=0.1, ridge_1=0.4)
    mean, se, _ = paired_difference_tstat(per_date["ridge_1"], per_date["rf_default"])
    t = mean / se
    lo = one_se_choice(rows, per_date, default_name="rf_default", se_threshold=t * 0.5)[0]
    hi = one_se_choice(rows, per_date, default_name="rf_default", se_threshold=t * 2.0)[0]
    assert lo == "ridge_1" and hi == "rf_default"


# ---- ensemble ----------------------------------------------------------------

def test_ensemble_uses_training_stats_only() -> None:
    frame = _panel("2019-01-01", "2020-12-31")
    train = frame[frame["date"] < "2020-07-01"]
    test = frame[frame["date"] >= "2020-07-01"]
    members = [(c["name"], fit_estimator(c["kind"], c["params"], train[FEATURES], train["target_return_1m"]))
               for c in (RIDGE, RF)]
    ens = StandardizedEnsemble(members, train[FEATURES])
    expected = np.zeros(len(test))
    for (_, m), (mu, sd) in zip(members, ens.stats):
        tr = m.predict(train[FEATURES])
        assert mu == pytest.approx(tr.mean()) and sd == pytest.approx(tr.std())
        expected += (m.predict(test[FEATURES]) - mu) / sd
    np.testing.assert_allclose(ens.predict(test[FEATURES]), expected / 2)
    # Predicting a subset must not change per-row outputs (no test-batch stats).
    np.testing.assert_allclose(ens.predict(test[FEATURES].iloc[:5]), ens.predict(test[FEATURES])[:5])


def test_select_model_nested_ensemble_rule() -> None:
    train = _panel("2018-01-01", "2022-10-31")
    model, meta = select_model_nested(
        train, feature_columns=FEATURES, target_column="target_return_1m",
        candidates=[RIDGE, RF], min_train_years=2, long_count=2, short_count=2,
        selection_rule="ensemble_equal",
    )
    assert isinstance(model, StandardizedEnsemble)
    assert meta["selection_mode"] == "ensemble_equal"
    assert model.names == ["ridge_1", "rf_default"]


def test_select_model_nested_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError):
        select_model_nested(
            _panel("2018-01-01", "2021-12-31"), feature_columns=FEATURES,
            target_column="target_return_1m", candidates=[RIDGE], selection_rule="oracle",
        )


# ---- HTB in inner scoring ----------------------------------------------------

def test_net_return_charges_name_level_borrow() -> None:
    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0, "F": -2.0})
    flat_rets = pd.Series(0.0, index=scores.index)
    kw = dict(previous_weights=None, long_count=2, short_count=2, gross_exposure=2.0,
              transaction_cost=0.0, annual_borrow_bps=100.0, annual_short_rebate_bps=0.0,
              annual_cash_financing_bps=0.0, periods_per_year=12)
    gc, w = net_long_short_period_return(scores, flat_rets, **kw)
    rates = pd.Series(100.0, index=scores.index)
    rates[w[w < 0].index[0]] = 3000.0  # one short is "special"
    htb, _ = net_long_short_period_return(scores, flat_rets, **kw, borrow_bps_by_symbol=rates)
    short_w = -w[w < 0].iloc[0]
    assert gc - htb == pytest.approx(short_w * (3000.0 - 100.0) / 10_000 / 12)


def _borrow_panel(train: pd.DataFrame, switch_date: str, special_ticker: str) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(train["date"].unique()))
    tickers = sorted(train["ticker"].unique())
    panel = pd.DataFrame(100.0, index=dates, columns=tickers)
    panel.loc[panel.index >= pd.Timestamp(switch_date), special_ticker] = 3000.0
    return panel


def test_inner_htb_panel_is_used_as_of_each_date() -> None:
    train = _panel("2018-01-01", "2022-12-31")
    common = dict(feature_columns=FEATURES, target_column="target_return_1m", candidate=RIDGE,
                  min_train_years=2, long_count=2, short_count=2, vol_targeting=False)
    flat = collect_inner_fold_diagnostics(train, **common)
    # T0 has the lowest feature → usually shorted. Make it special only from 2022.
    panel = _borrow_panel(train, "2022-01-01", "T0")
    htb = collect_inner_fold_diagnostics(train, **common, borrow_bps_panel=panel)
    for date, net in flat["net_by_date"].items():
        if date < pd.Timestamp("2022-01-01"):
            # Panel = 100 bps = flat GC before the switch → identical nets.
            assert htb["net_by_date"][date] == pytest.approx(net)
    later = [d for d in flat["net_by_date"] if d >= pd.Timestamp("2022-01-01")]
    assert later
    assert sum(htb["net_by_date"][d] for d in later) < sum(flat["net_by_date"][d] for d in later)


def test_inner_htb_ignores_panel_rows_after_train_window() -> None:
    frame = _panel("2018-01-01", "2023-12-31")
    train = frame[frame["date"] < "2022-11-15"]
    panel_a = _borrow_panel(frame, "2099-01-01", "T0")  # all GC
    panel_b = panel_a.copy()
    panel_b.loc[panel_b.index >= pd.Timestamp("2023-01-01")] = 1e6  # poison outer-test rows
    kwargs = dict(feature_columns=FEATURES, target_column="target_return_1m",
                  candidates=[RIDGE, RF], min_train_years=2, long_count=2, short_count=2)
    _, meta_a = select_model_nested(train, borrow_bps_panel=panel_a, **kwargs)
    _, meta_b = select_model_nested(train, borrow_bps_panel=panel_b, **kwargs)
    assert meta_a["inner_name_level_borrow"] is True
    for a, b in zip(meta_a["candidate_scores"], meta_b["candidate_scores"]):
        assert a["inner_score"] == pytest.approx(b["inner_score"])
    assert meta_a["chosen_model"] == meta_b["chosen_model"]
