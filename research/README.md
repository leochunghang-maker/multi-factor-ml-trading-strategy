# research/

Research bookkeeping and one-off diagnostics. Nothing here trades, downloads data or
changes a committed result by itself.

| File | What it is |
|---|---|
| `trial_log.csv` | The honest count of research variants tried. Feeds `n_trials` for the Deflated Sharpe Ratio. |
| `diagnose_multi_factor_beta.py` | A descriptive diagnosis of why the beta-neutral multi-factor book had negative realized beta (`python research/diagnose_multi_factor_beta.py`, step `beta-diagnosis` in `make research`; writes `results/multi_factor_beta_diagnosis.csv`). It chooses no parameter. Quantities that use look-ahead are prefixed `ex_post_` and used only for diagnosis. |

## Why a trial log

Each variant we look at is another draw. The best of many skill-less draws will look good by
chance, so a Sharpe ratio has to be judged against how many variants were tried. The Deflated
Sharpe Ratio (DSR; Bailey & López de Prado 2014) does this by comparing the observed Sharpe with
SR0, the expected maximum Sharpe of `N` skill-less trials. If `N` is undercounted, the DSR is
inflated. The log keeps `N` honest and auditable in git.

## Columns

`trial_id,date,commit,family,description,trial_increment,reported_sharpe,notes`

- `trial_id`: the next integer. Never reuse one, never renumber.
- `date`: the day the row was written (YYYY-MM-DD).
- `commit`: the commit that contains the variant. Use `pending` until it is committed, then fill in
  the short hash.
- `family`: the series the trial belongs to. Its logged Sharpes give the cross-trial variance
  V[SR] for that family: `walk_forward_ml`, `multi_factor`, `momentum`, `ml_baseline`,
  `nested_candidate`, `multi_day_sim`, `benchmark_relative`.
- `description`: the variant, specific enough to reproduce.
- `trial_increment`: how many trials the row adds to `N`. Usually `1`.
- `reported_sharpe`: the annualized Sharpe once observed. Leave it empty if none was observed.
- `notes`: say `Logged before running` when the row was pre-logged, plus anything else a reviewer
  needs.

`src/reporting/trial_log.py` validates the file. It rejects a missing column, a duplicate
`trial_id`, or a negative or non-integer increment.

## Discipline

1. **Pre-log.** Append the row before you run a new variant, sensitivity or test whose result
   you will see. Commit it with (or before) the code. Fix the design first (parameters, sample,
   metric), not after seeing numbers.
2. **Never delete or edit away rows** for variants that did not work. A failed variant still counts.
   Later corrections go in `notes` or in a new row.
3. **Increment rules:**
   - Use `1` for a new variant, sensitivity run or test whose result is observed. This applies
     even if it is not adopted, and even if it is "just a check".
   - Use `0` when nothing new is searched: regenerating a stale artifact with unchanged code, or an
     engineering bug fix that changes committed numbers without choosing anything. Log the row
     anyway, so the change in numbers is explained (example: row 44).
   - Use one row for one design decision applied consistently in several places (example: row 43,
     the EWMA beta estimator applied to the multi-factor and walk-forward books), with the details
     in `notes`.
   - Pure engineering with no research or report number affected (paper-ops tooling, tests,
     refactors) is not logged.
4. **Do not tune on the result.** Thresholds and settings fixed a priori stay fixed even if the
   result disappoints. Report the result honestly in README and reports.

## How N is computed

```
N = Σ trial_increment  +  significance.trial_padding      (config/platform_config.json)
```

The fixed, documented padding covers unrecorded pre-2026-11 iterations and researcher choices
that were never swept: long/short counts, lookbacks, target vol, half-lives and thresholds.
Because `N` is probably still an undercount, `results/significance_report.csv` also reports DSR at
the sensitivity values in `significance.n_trials_sensitivity` (15, 50, 100, 200). V[SR] is the
variance of each family's logged `reported_sharpe`. A pooled variance over all logged Sharpes is
reported as a more conservative check.

The current values are quoted (and checked) in the README section "Trial log and the number of
trials". To print them:

```bash
python -c "from src.config import SIGNIFICANCE_TRIAL_LOG_PATH as p, SIGNIFICANCE_TRIAL_PADDING as k; \
from src.reporting.trial_log import load_trial_log, count_trials; \
log = load_trial_log(p); print(int(log.trial_increment.sum()), '+', k, '=', count_trials(log, k))"
```

## How to pre-log a trial

1. Append a row. Example:

   ```
   45,2026-11-10,pending,multi_factor,<exact variant>,1,,Logged before running
   ```

2. Commit the row with the code change, then run the variant (for example `make research`, or
   `python scripts/run_research_pipeline.py --from multi-factor`).
3. Fill in `reported_sharpe` and replace `pending` with the commit hash.
4. A row with a non-zero increment changes `N`, which changes every DSR. Regenerate the
   significance report (`python src/run_significance_report.py --cached-walk-forward`, or the
   `significance` step of `make research`). Then refresh the quoted numbers with
   `python src/check_doc_metrics.py --fix` and review the diff. `pytest` (`tests/test_doc_metrics.py`)
   and `python src/check_doc_metrics.py` fail until README and reports match the result files.
5. Describe the result in README/reports whether it is better or worse. Do not revert a logged
   trial because of its outcome.
