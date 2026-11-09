"""Data freshness, alignment, quality checks and the checked-in data manifest.

Read-only: nothing here modifies a data file. Findings carry a severity:

- ``ERROR``   — the data is unusable or inconsistent (smoke tests fail).
- ``WARNING`` — needs a human look (stale snapshot, empty column, split-like jump,
  file changed since the manifest).
- ``INFO``    — recorded for the report only (late listings, large but not
  split-like moves).

Settings live in ``config/platform_config.json`` → ``data_checks``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEVERITIES = ("ERROR", "WARNING", "INFO")
SPLIT_FACTORS = (1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 20.0)

DEFAULT_DATA_CHECKS: dict[str, Any] = {
    "max_stale_trading_days": 5,
    "stale_severity": "WARNING",
    "max_end_date_gap_trading_days": 0,
    "max_missing_aligned_dates": 0,
    "misalignment_severity": "ERROR",
    "max_calendar_gap_days": 4,
    "max_missing_cross_section_fraction": 0.5,
    "flat_run_min_days": 5,
    "large_move_threshold": 0.25,
    "split_move_threshold": 0.4,
    "split_ratio_tolerance": 0.03,
    "manifest_path": "data/data_manifest.json",
    "manifest_files": [
        "data/price_data.csv",
        "data/volume_data.csv",
        "data/benchmark_prices.csv",
    ],
    "hash_mismatch_severity": "WARNING",
    "excluded_symbols": {},
    "paper_stale_severity": "ERROR",
    "report_path": "reports/data_quality_report.md",
}

# The three time-series inputs checked for freshness and alignment.
DATED_INPUTS = {
    "price_data": "data/price_data.csv",
    "volume_data": "data/volume_data.csv",
    "benchmark_prices": "data/benchmark_prices.csv",
}


@dataclass
class DataFinding:
    severity: str
    check: str
    file: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")


def data_check_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Defaults overlaid with ``config['data_checks']`` (platform config if None)."""
    if config is None:
        from src.operations.configuration import load_platform_config

        config = load_platform_config()
    return {**DEFAULT_DATA_CHECKS, **(config.get("data_checks") or {})}


# --------------------------------------------------------------------------- io


def load_dated_csv(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    return frame


def trading_days_between(start, end) -> int:
    """Weekdays strictly after ``start`` up to and including ``end`` (holidays not excluded)."""
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    if end <= start:
        return 0
    return int(np.busday_count(start + pd.Timedelta(days=1), end + pd.Timedelta(days=1)))


# -------------------------------------------------------------------- freshness


def check_freshness(
    frames: dict[str, pd.DataFrame | None],
    as_of,
    settings: dict[str, Any],
    reference: str = "price_data",
) -> list[DataFinding]:
    """Staleness of each dated input and alignment of their date ranges/index sets."""
    findings: list[DataFinding] = []
    max_stale = int(settings["max_stale_trading_days"])
    stale_sev = str(settings["stale_severity"])
    mis_sev = str(settings["misalignment_severity"])
    as_of = pd.Timestamp(as_of).normalize()
    for name, frame in frames.items():
        if frame is None or frame.empty:
            findings.append(DataFinding("ERROR", "freshness", name, "file missing or empty"))
            continue
        last = frame.index.max()
        age = trading_days_between(last, as_of)
        if age > max_stale:
            findings.append(DataFinding(
                stale_sev, "freshness", name,
                f"last date {last.date()} is {age} trading days before {as_of.date()} (limit {max_stale})",
            ))
    ref = frames.get(reference)
    if ref is None or ref.empty:
        return findings
    ref_index = pd.DatetimeIndex(ref.index)
    for name, frame in frames.items():
        if name == reference or frame is None or frame.empty:
            continue
        idx = pd.DatetimeIndex(frame.index)
        end_gap = trading_days_between(min(idx.max(), ref_index.max()), max(idx.max(), ref_index.max()))
        start_gap = trading_days_between(min(idx.min(), ref_index.min()), max(idx.min(), ref_index.min()))
        if end_gap > int(settings["max_end_date_gap_trading_days"]):
            findings.append(DataFinding(
                mis_sev, "alignment", name,
                f"ends {idx.max().date()} but {reference} ends {ref_index.max().date()} ({end_gap} trading days apart)",
            ))
        if start_gap > int(settings["max_end_date_gap_trading_days"]):
            findings.append(DataFinding(
                mis_sev, "alignment", name,
                f"starts {idx.min().date()} but {reference} starts {ref_index.min().date()} ({start_gap} trading days apart)",
            ))
        lo, hi = max(idx.min(), ref_index.min()), min(idx.max(), ref_index.max())
        ref_in = ref_index[(ref_index >= lo) & (ref_index <= hi)]
        own_in = idx[(idx >= lo) & (idx <= hi)]
        missing = ref_in.difference(own_in)
        extra = own_in.difference(ref_in)
        limit = int(settings["max_missing_aligned_dates"])
        if len(missing) > limit or len(extra) > limit:
            sample = ", ".join(str(d.date()) for d in list(missing[:3]) + list(extra[:3]))
            findings.append(DataFinding(
                mis_sev, "alignment", name,
                f"{len(missing)} {reference} dates missing and {len(extra)} extra dates in the overlap (e.g. {sample})",
            ))
    return findings


# ---------------------------------------------------------------------- quality


def _runs_of_true(mask: pd.Series) -> pd.Series:
    """Length of each consecutive run of True values (indexed by run end)."""
    groups = (~mask).cumsum()
    lengths = mask.groupby(groups).sum()
    ends = mask.groupby(groups).apply(lambda s: s.index[-1])
    return pd.Series(lengths.to_numpy(), index=pd.DatetimeIndex(ends.to_numpy()))


def split_like_factor(price_ratio: float, tolerance: float) -> float | None:
    """Return k if ``price_ratio`` ≈ 1/k or k for a common split factor, else None."""
    if not np.isfinite(price_ratio) or price_ratio <= 0:
        return None
    for k in SPLIT_FACTORS:
        for target in (1.0 / k, k):
            if abs(price_ratio / target - 1.0) <= tolerance:
                return k
    return None


def check_panel_quality(
    frame: pd.DataFrame | None,
    name: str,
    settings: dict[str, Any],
    *,
    kind: str = "price",
) -> list[DataFinding]:
    """Structural and value checks on a date × symbol panel (``kind``: price|volume)."""
    if frame is None:
        return [DataFinding("ERROR", "missing_file", name, "file missing")]
    findings: list[DataFinding] = []
    idx = frame.index
    if idx.isna().any():
        findings.append(DataFinding("ERROR", "dates", name, f"{int(idx.isna().sum())} unparseable dates"))
        frame = frame.loc[idx.notna()]
        idx = frame.index
    if idx.duplicated().any():
        findings.append(DataFinding("ERROR", "dates", name, f"{int(idx.duplicated().sum())} duplicate dates"))
    if not idx.is_monotonic_increasing:
        findings.append(DataFinding("ERROR", "dates", name, "dates are not sorted ascending"))
    weekend = idx[idx.day_of_week >= 5]
    if len(weekend):
        findings.append(DataFinding("WARNING", "dates", name, f"{len(weekend)} weekend dates (e.g. {weekend[0].date()})"))

    # Missing rows: calendar gaps longer than a normal holiday weekend.
    gaps = pd.Series(idx[1:] - idx[:-1], index=idx[1:]).dt.days if len(idx) > 1 else pd.Series(dtype=float)
    long_gaps = gaps[gaps > int(settings["max_calendar_gap_days"])]
    for end, days in long_gaps.items():
        findings.append(DataFinding("WARNING", "missing_rows", name, f"{int(days)}-day gap ending {end.date()}"))

    values = frame.apply(pd.to_numeric, errors="coerce")
    empty = [str(c) for c in values.columns if values[c].isna().all()]
    excluded = dict(settings.get("excluded_symbols") or {})
    decided = [c for c in empty if c in excluded]
    undecided = [c for c in empty if c not in excluded]
    if undecided:
        findings.append(DataFinding("WARNING", "empty_columns", name, f"all-NaN column(s): {', '.join(undecided)}"))
    if decided:
        findings.append(DataFinding(
            "INFO", "excluded_symbols", name,
            f"all-NaN by recorded decision (data_checks.excluded_symbols): {', '.join(decided)}",
        ))
    not_empty = [c for c in excluded if c in values.columns and c not in empty]
    if not_empty:
        findings.append(DataFinding(
            "WARNING", "excluded_symbols", name,
            f"excluded symbol(s) have data and could enter the universe: {', '.join(not_empty)}",
        ))
    active = values.drop(columns=empty)
    if active.empty:
        return findings

    first = active.apply(lambda s: s.first_valid_index())
    last = active.apply(lambda s: s.last_valid_index())
    late = first[first > idx.min()]
    if len(late):
        listed = ", ".join(f"{c} {d.date()}" for c, d in late.sort_values().items())
        findings.append(DataFinding("INFO", "coverage", name, f"history starts late: {listed}"))
    early = last[last < idx.max()]
    if len(early):
        listed = ", ".join(f"{c} {d.date()}" for c, d in early.sort_values().items())
        findings.append(DataFinding("WARNING", "coverage", name, f"history ends early: {listed}"))

    in_life = pd.DataFrame(
        {c: (idx >= first[c]) & (idx <= last[c]) for c in active.columns}, index=idx
    )
    holes = active.isna() & in_life
    hole_counts = holes.sum()
    hole_counts = hole_counts[hole_counts > 0]
    if len(hole_counts):
        listed = ", ".join(f"{c} ({int(n)})" for c, n in hole_counts.sort_values(ascending=False).items())
        findings.append(DataFinding("WARNING", "missing_values", name, f"NaN inside listed history: {listed}"))
    frac = holes.sum(axis=1) / in_life.sum(axis=1).replace(0, np.nan)
    thin = frac[frac > float(settings["max_missing_cross_section_fraction"])]
    if len(thin):
        findings.append(DataFinding(
            "WARNING", "missing_rows", name,
            f"{len(thin)} dates with >{settings['max_missing_cross_section_fraction']:.0%} of listed names missing "
            f"(e.g. {thin.index[0].date()})",
        ))

    if kind == "volume":
        negative = int((active < 0).sum().sum())
        if negative:
            findings.append(DataFinding("ERROR", "values", name, f"{negative} negative volumes"))
        zeros = (active == 0).sum()
        zeros = zeros[zeros > 0]
        if len(zeros):
            listed = ", ".join(f"{c} ({int(n)})" for c, n in zeros.items())
            findings.append(DataFinding("WARNING", "values", name, f"zero-volume days: {listed}"))
        return findings

    nonpositive = int((active <= 0).sum().sum())
    if nonpositive:
        findings.append(DataFinding("ERROR", "values", name, f"{nonpositive} non-positive prices"))

    min_run = int(settings["flat_run_min_days"])
    for col in active.columns:
        series = active[col].dropna()
        if len(series) < 2:
            continue
        runs = _runs_of_true(series.diff().eq(0))
        long_runs = runs[runs + 1 >= min_run]  # n unchanged diffs = n + 1 identical closes
        for end, length in long_runs.items():
            findings.append(DataFinding(
                "WARNING", "flat_prices", name,
                f"{col}: {int(length) + 1} identical closes ending {end.date()} (stale/forward-filled?)",
            ))

    returns = active.pct_change(fill_method=None)
    large = float(settings["large_move_threshold"])
    split_move = float(settings["split_move_threshold"])
    tol = float(settings["split_ratio_tolerance"])
    stacked = returns.stack()
    for (when, col), ret in stacked[stacked.abs() >= large].items():
        ratio = 1.0 + float(ret)
        k = split_like_factor(ratio, tol) if abs(ret) >= split_move else None
        if k is not None:
            findings.append(DataFinding(
                "WARNING", "split_like_jump", name,
                f"{col} {when.date()}: {ret:+.1%} (price ratio ≈ {'1/' if ratio < 1 else ''}{k:g}; likely unadjusted split)",
            ))
        else:
            findings.append(DataFinding(
                "INFO", "large_move", name, f"{col} {when.date()}: {ret:+.1%} (not split-like)",
            ))
    return findings


def check_volume_vs_price(prices: pd.DataFrame | None, volumes: pd.DataFrame | None) -> list[DataFinding]:
    if prices is None or volumes is None:
        return []
    common_cols = prices.columns.intersection(volumes.columns)
    common_idx = prices.index.intersection(volumes.index)
    p = prices.loc[common_idx, common_cols]
    v = volumes.loc[common_idx, common_cols]
    missing = int((p.notna() & v.isna()).sum().sum())
    only_p = sorted(set(prices.columns) - set(volumes.columns))
    out = []
    if missing:
        out.append(DataFinding("WARNING", "missing_values", "volume_data", f"{missing} cells with a price but no volume"))
    if only_p:
        out.append(DataFinding("WARNING", "coverage", "volume_data", f"symbols without volume column: {', '.join(only_p)}"))
    return out


# --------------------------------------------------------------------- manifest


def file_sha256(path: str | Path) -> str:
    """SHA-256 of the file bytes with CRLF normalised to LF (checkout-independent)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def describe_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    entry: dict[str, Any] = {"sha256": file_sha256(path)}
    try:
        frame = pd.read_csv(path)
        entry["rows"] = int(len(frame))
        entry["columns"] = int(frame.shape[1])
        first_col = pd.to_datetime(frame.iloc[:, 0], errors="coerce", format="mixed")
        if len(frame) and first_col.notna().mean() > 0.9:
            entry["first_date"] = str(first_col.min().date())
            entry["last_date"] = str(first_col.max().date())
    except Exception:  # pragma: no cover - non-CSV or unreadable
        pass
    return entry


def build_manifest(
    files: list[str],
    root: str | Path = ".",
    decisions: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    entries = {f: describe_file(root / f) for f in files if (root / f).exists()}
    manifest = {
        "description": "SHA-256 (CRLF→LF) and shape of checked-in data inputs. Regenerate only after "
        "reviewing a re-download: python src/check_data.py --update-manifest",
        "updated": date.today().isoformat(),
        "files": entries,
    }
    if decisions:
        manifest["excluded_symbols"] = dict(decisions)
    return manifest


def blank_excluded_symbols(frame: pd.DataFrame, excluded) -> pd.DataFrame:
    """Set excluded symbols to NaN (adding the column if absent), keeping sorted columns.

    Used after downloads so a refresh cannot silently bring an excluded name
    into the research or paper universe.
    """
    out = frame.copy()
    for symbol in excluded or []:
        out[symbol] = np.nan
    return out.reindex(columns=sorted(out.columns))


def check_manifest(
    manifest_path: str | Path,
    files: list[str],
    settings: dict[str, Any],
    root: str | Path = ".",
) -> list[DataFinding]:
    root = Path(root)
    manifest_path = root / manifest_path
    if not manifest_path.exists():
        return [DataFinding("WARNING", "manifest", str(manifest_path), "no data manifest; run --update-manifest")]
    manifest = json.loads(manifest_path.read_text())
    recorded = manifest.get("files", {})
    severity = str(settings["hash_mismatch_severity"])
    findings = []
    for f in files:
        path = root / f
        if not path.exists():
            findings.append(DataFinding("ERROR", "manifest", f, "listed in manifest_files but missing on disk"))
            continue
        if f not in recorded:
            findings.append(DataFinding("WARNING", "manifest", f, "not recorded in the data manifest"))
            continue
        now = describe_file(path)
        if now["sha256"] != recorded[f].get("sha256"):
            was = recorded[f]
            detail = (
                f"rows {was.get('rows')}→{now.get('rows')}, last date {was.get('last_date')}→{now.get('last_date')}"
            )
            findings.append(DataFinding(
                severity, "manifest", f,
                f"content changed since manifest ({detail}); review, re-run `make research`, then --update-manifest",
            ))
    return findings


# ------------------------------------------------------------------------ driver


def run_data_checks(
    as_of=None,
    config: dict[str, Any] | None = None,
    root: str | Path = ".",
) -> tuple[list[DataFinding], dict[str, Any]]:
    """All freshness, alignment, quality and manifest checks. Returns findings + summary."""
    settings = data_check_settings(config)
    root = Path(root)
    as_of = pd.Timestamp(as_of or date.today()).normalize()
    frames = {name: load_dated_csv(root / path) for name, path in DATED_INPUTS.items()}
    findings: list[DataFinding] = []
    findings += check_freshness(frames, as_of, settings)
    findings += check_panel_quality(frames["price_data"], "price_data", settings, kind="price")
    findings += check_panel_quality(frames["volume_data"], "volume_data", settings, kind="volume")
    findings += check_panel_quality(frames["benchmark_prices"], "benchmark_prices", settings, kind="price")
    findings += check_volume_vs_price(frames["price_data"], frames["volume_data"])
    findings += check_manifest(settings["manifest_path"], list(settings["manifest_files"]), settings, root=root)
    summary = {
        "as_of": str(as_of.date()),
        "files": {
            name: (
                {"rows": int(len(f)), "columns": int(f.shape[1]),
                 "first_date": str(f.index.min().date()), "last_date": str(f.index.max().date())}
                if f is not None and not f.empty else None
            )
            for name, f in frames.items()
        },
        "counts": {s: sum(f.severity == s for f in findings) for s in SEVERITIES},
    }
    return findings, summary


def overall_status(findings: list[DataFinding]) -> str:
    if any(f.severity == "ERROR" for f in findings):
        return "ERROR"
    if any(f.severity == "WARNING" for f in findings):
        return "WARNING"
    return "OK"


def findings_to_frame(findings: list[DataFinding]) -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in findings], columns=["severity", "check", "file", "message"])
