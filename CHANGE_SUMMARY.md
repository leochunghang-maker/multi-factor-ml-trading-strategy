# Change Summary

What changed in this repository, grouped by area, as of `main` on 2026-11-09. Commit hashes are
given so each item can be checked. Current performance numbers live in `README.md` and
`reports/quant_research_report.md`. They are checked against the result files by
`python src/check_doc_metrics.py`, so this file does not repeat them.

**Bottom line.** None of the strategies shows statistically significant alpha. Every HAC alpha
t-stat is below 1.96, and every deflated IR or hedged Deflated Sharpe is far below 0.95. Paper
trading is local and simulated only. There is no live-money execution path.

## Research correctness (Nov 2026)

- **Walk-forward ML** now uses month-end, non-overlapping holding-period returns
  (`069d595`). The earlier daily overlapping 21-day returns overstated results. Training uses
  purged and embargoed splits (`486f202`). Full-history data and metrics were regenerated
  (`ed838d3`, `7776399`).
- Research long/short gross exposure is explicit (2.0) and separate from the paper cap of 1.0
  (`7fcdf2d`).
- **Costs:** half-spread and square-root market impact (`6ad2d61`). Short borrow and financing
  (`6152dd3`), plus name-level hard-to-borrow fee tiers from an ADV/volatility proxy (`8173d69`).
  An ADV participation cap applies to research and paper/sim (`d385979`). Volume data comes from
  Yahoo, so the ADV filter is active (`619b9a0`).
- **Universe:** point-in-time eligibility (tradability plus a month-end S&P 500 membership
  overlay) reduces survivorship bias without removing it (`530d8bd`). BK is excluded; the
  decision is recorded with the loader's `--end`/`--with-benchmark` options (`5627193`).
- **Model selection:** nested anchored walk-forward (`f761cd4`). Inner scoring is cost-aware and
  consistent with the outer book, with an ICIR option (`40c2ab1`, `2c6d681`). A one-SE selection
  rule and name-level borrow in inner scoring (`147723c`).
- **Construction:** sector-neutral long/short with net exposure caps (`9ab703f`). Volatility
  targeting, realized or EWMA (`6369768`, `b517457`). Beta-neutral legs by construction
  (`fe8b173`), then an EWMA beta estimator, plus a descriptive diagnosis of the multi-factor
  book's residual beta (`1d81457`, `research/diagnose_multi_factor_beta.py`).
- **Significance:** stationary block-bootstrap Sharpe CI, PSR and Deflated Sharpe (`c6dd1e6`).
  Benchmark-relative alpha, beta, IR and hedged tests against SPY and the equal-weight universe
  (`e433b15`).
- **Trial log:** `research/trial_log.csv` is checked in and feeds the DSR's number of trials,
  `N = Σ trial_increment + padding` (`e433b15`). Rows are pre-logged before running. The rules
  are in `research/README.md` (`19e251b`).
- **Reproducibility:** `make research` regenerates every committed artifact in dependency order.
  Numbers quoted in the README and reports carry markers, and a test fails if they drift from
  the result files (`7ca79e0`, `650855a`).
- **Data checks:** freshness on the NYSE calendar, alignment, quality (flat runs, large or split
  moves) and a checked-in data manifest with hashes (`1b26493`, `018b99b`).

## Simulated paper trading (Nov 2026)

- Drawdown circuit breaker: the rebalance is frozen at or below the threshold (`75796fa`).
- **Operations:**
  - idempotent runs keyed by signal date, with dated snapshots, start-state reconciliation, a
    journal and a pre-run data gate where staleness is an ERROR (`10b7066`)
  - a separate `data/paper/` directory, so the research snapshot is never rewritten (`018b99b`)
  - alerts and a status summary (`60947ef`)
  - backup and restore with integrity checks, and quiet refreshes (`8b07352`)
  - `make paper-daily` with a run lock, rotating backups and a weekly review (`a961a48`)
  - a read-only Paper Account tab in the dashboard (`192952b`)
- **Tracking check:** each run is compared with a frictionless backtest of the same signals
  (holding gap, cost ratio, weight deviation). Thresholds were fixed a priori. Breaches raise
  alerts and appear in status and the weekly review (`754bdc3`).
- **Closed sessions only:** the paper refresh downloads through the last NYSE session whose
  New York close + 30 min has passed, and the data gate blocks any row dated after it. A run
  during US market hours used to trade and mark the book at a partial intraday bar and then skip
  the after-close rerun for that date. Freshness is measured from the last closed session in New
  York time instead of the local date. Research numbers are unaffected.
- **Rebalance bug fix:** sells now execute before buys, and buys are sized from the cash left
  after the sells. Before this, exits listed after the buys caused insufficient-cash rejections
  (`5102ebe`). No checked-in number changed.

## Engineering and robustness

- Config: risk, execution and simulation settings come from `config/platform_config.json`
  (`2d89585`). `tests/test_config_keys.py` checks that keys read in `src/` and keys in the file
  match (`e781fe7`).
- A missing input fails with one line naming the file and the command that creates it, with exit
  code 2 (`src/data/required.py`, `e781fe7`).
- Dependencies: `requirements.txt` lists the direct dependencies. `requirements.lock` pins
  everything with hashes, and CI installs from it (`dabf444`, `733338b`).
- CI runs pytest and the smoke tests on every push to `main` (`e468f4e`). `pytest.ini` turns
  warnings into failures, and the suite has 0 warnings (`8d9e481`).

## Original structural refactor (May 2026)

- `src/config.py` centralizes the universe, paths, features, split dates, costs and risk
  settings. Shared modules: `src/data/io.py`, `src/features/factors.py`, `src/models/ml.py`,
  `src/portfolio/portfolio.py`, `src/reporting/{metrics,analytics,charts,summary}.py`,
  `src/backtests/{factor_backtests,ml_backtests,benchmark}.py` and `src/risk/volatility.py`.
- Scripts became import-safe `main()` entry points. The embedded ML block was removed from
  `src/multi_factor_backtest.py`. Compatibility wrappers remain in `src/metrics.py` and
  `src/portfolio.py`.
- The first platform layer was built: paper-trading simulator, risk analytics, portfolio
  constraints, monitoring, dashboard and docs (`6550f39`, `ebd8ba8`, `0bd6275`).
- The original May narrative figures (23.1% / 24.92% annual) are not reproducible. They are
  logged as trials 1–2 and described in the README's "Older narrative snapshot".
