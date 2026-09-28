# Chapter 6 — Integrity and Process

This chapter exists to help you defend AlphaBERT in front of judges who are actively hunting for
look-ahead bias, survivorship bias and leakage. The brief is explicit about this (p.12: *"The
reproducibility agent... whose job is to re-run your own pipeline and hunt for look-ahead bias,
survivorship and leakage before you submit. We will be checking; you should check first."*, and
p.17: *"Teams that cannot explain how they ruled out model-side look-ahead will be treated as
having used it."*). Every claim below is grounded in a specific file, function or test you can
point a judge to and re-run live. Where the project has a genuine weak spot, this chapter says so
plainly — the brief itself rewards candor over polish (p.12: *"A candid account of an agent that
did not work is worth more to us than a polished account of one that supposedly did."*).

All file references are relative to the repo root. All test names are exact and were verified
against the current tree; the fast (`not slow`) suite was executed for this chapter
(`python -m pytest -q -m "not slow"` → **88 passed in ~58s**, 2026-09-28). The `slow` suite (19
tests, real parquet reads + LightGBM fits) was not re-executed for this chapter — its runtime is
described below from the tests' own docstrings, not remeasured.

---

## 6.1 Threat model

Thirteen distinct ways look-ahead, survivorship or leakage could plausibly enter this pipeline,
and the specific code-level defense and test for each.

| # | Threat | How it could enter this pipeline | Defense in code | Proof (test) |
|---|---|---|---|---|
| 1 | **Target shifting** | Using `stock_exret` (or anything derived from it) as a feature, or re-shifting an already-shifted return | `src/data.py::_build` sets `stock_exret = ret_exc_lead1m` exactly once, no further shift; `target_month = eom + MonthEnd(1)` is the only place this is computed. `models.py::_FORBIDDEN = {"stock_exret", "target_month", "ret_exc_lead1m"}` and `_assert_no_leakage()` run against every feature list before every `run_all()` fit. | `tests/test_data.py::test_target_month_and_stock_exret`, `test_feature_columns_excludes_target_derived`; `tests/test_integrity.py::test_target_alignment_real` (checks `stock_exret(t) == raw ret_exc(t+1)` on a 5,000-row real sample, match rate >99.9%), `test_feature_hygiene_real` |
| 2 | **Universe conditioned on future returns** | Filtering the investable universe using `ret_exc_lead1m` (e.g. dropping stocks that are about to delist) would build survival into the sample selection itself | `src/data.py::universe_mask` uses only `prc >= MIN_PRICE` and `me >=` the cross-sectional quantile — it never reads `ret_exc_lead1m` | `tests/test_data.py::test_universe_mask_ignores_ret_exc_lead1m` (mask unchanged whether `ret_exc_lead1m` is all-NaN, real, or randomized); `tests/test_integrity.py::test_universe_invariance_to_nan_ret_exc_lead1m_real` (real data, 1/3 of rows NaN'd) |
| 3 | **Preprocessing fit on future data** | Cross-sectional ranks, `miss_` flags, `market_state()`, and text features computed for month *t* accidentally using information from *t+1..T* (e.g. a global z-score over the whole sample) | Ranks (`_rank_within_eom`) and `size_z` are computed strictly `groupby('eom')`; `market_state()`'s rolling windows (`mkt_ret12`, `mkt_vol12`) use `.rolling(12, min_periods=6)` ending at *t*; text features key on `filing_date`'s own month | `tests/test_integrity.py::test_truncation_invariance_panel_real`, `test_truncation_invariance_market_state_real`, `test_truncation_invariance_text_real` (rebuild each artifact from data truncated at `T=2021-06-30` and show month-*T* output is byte-identical to the full build); synthetic counterparts `tests/test_data.py::test_market_state_truncation_invariance`, `test_real_panel_truncation_invariance`, `test_real_market_state_truncation_invariance`, `tests/test_text.py::test_truncation_invariance` |
| 4 | **Missing-flag selection using future data** | Deciding which characteristics get a `miss_<c>` flag using the *full* sample's missing rate (rather than a fixed historical cutoff) would let a flag's very existence depend on future data patterns | `src/data.py::_build` computes `miss_rate` only over universe rows with `eom <= config.MISS_FLAG_CUTOFF` (2018-12-31), then applies the resulting flag list to all rows | `tests/test_data.py::test_miss_flags_use_only_cutoff_rows` (a char missing only *after* the cutoff must not get flagged) |
| 5 | **Validation/test overlap in fitting or tuning** | Any specialist, baseline or the gate touching valid or test labels beyond what SPEC section 2 allows | `models.py::splits()` requires a non-null label for train, not for valid/test; `_fit_specialist()` tunes `num_leaves`/rounds on an **inner holdout inside train** (last 12 target months), never on official valid (A2); the gate is the only model fit on valid, and only on its labelled subset | `tests/test_integrity.py::test_schedule_and_inner_holdout`; `tests/test_models.py::test_specialist_ignores_valid_labels`, `test_run_all_ignores_valid_labels` (corrupting valid labels changes only ridge/lasso/enet/gate, never the specialists/lgbm_all/OLS/pred_ew), `test_real_panel_ignores_test_labels` (**real data**: corrupting test-split `stock_exret` changes zero columns in `models.PRED_COLS`) |
| 6 | **Hyperparameter tuning on test** | Choosing `num_leaves`, boosting rounds, or ridge alpha by looking at test-period performance | Same defenses as #5 — nothing in `run_all()` ever reads a test-split label; `r2_table()` (which does read test labels) is called only *after* `run_all()` returns, on a separate DataFrame | `tests/test_models.py::test_real_panel_ignores_test_labels` (real-data proof that **every** column in `PRED_COLS` — baselines, all 6 specialists, both gates, both equal-weight blends — is bit-identical whether or not test labels are corrupted) |
| 7 | **Text filing dates (look-ahead via 8-K timing)** | A filing dated after month *t* contaminating month-*t* text features, or a filing's calendar-month assignment being computed inconsistently | `src/text.py::build_text_features` keys strictly on `eom = filing_date + MonthEnd(0)`; module docstring states the invariant: *"features at eom t use only filings with filing_date in calendar month t"* | `tests/test_text.py::test_build_text_features_month_assignment_and_counts` (a Feb-1 filing must not count in January), `test_truncation_invariance`; `tests/test_integrity.py::test_truncation_invariance_text_real` (real 8-K archive) |
| 8 | **Survivorship via filing coverage** | The 8-K archive is retrospectively assembled: same-month filing presence strongly predicts *future* survival, so using `has_filing` (or letting it leak into the text specialist) as a feature would smuggle in hindsight | A1: `text.filing_coverage_by_exit()` measured this directly (coverage 1.3% at last panel row → 59.7% for survivors to panel end); `has_filing` is excluded from `TEXT_FEATURES`, the text specialist trains/predicts on filer rows only, its within-month z-score treats non-filers as an exact 0 (`models._zscore_by_eom`), and the optimizer enforces a filer-net-neutral constraint (A10, `portfolio._tol_groups`) so the mechanically filer-skewed text signal can't tilt the book | `tests/test_text.py::test_filing_coverage_by_exit_buckets_and_censoring` + real `test_filing_coverage_by_exit_runs`; `tests/test_models.py::test_zscore_by_eom_zero_for_nan_rows`; `tests/test_portfolio.py::test_optimize_month_filer_net_neutral`, `test_backtest_filer_net_neutral_diagnostic`; `tests/test_integrity.py::test_mini_end_to_end_pipeline` (asserts `filer_net` stays within tolerance end-to-end) |
| 9 | **Stale caches** | A settings change (FinBERT `max_length`, cleaning revision, or a `models.py` code change) silently reusing an old cache and producing results that don't match the current code | `text.scores_path_for(max_length)` makes the FinBERT cache filename settings-specific; `build_text_features()` asserts the cached scores' `max_length`/`revision` match the current settings and that no `document_id` is missing; `MAIN.py --reuse-preds` refuses to reuse `preds.parquet` if it's missing any `PRED_COLS` or is older than `src/models.py`'s mtime | `tests/test_text.py::test_build_text_features_raises_when_scores_settings_mismatch[max_length-128]` and `[revision-some-other-commit]`, `test_build_text_features_raises_when_scores_missing_document_ids`, `test_chunk_dir_for_separates_max_lengths`. **Gap:** `MAIN.py`'s `--reuse-preds` mtime/column guard (`MAIN.py` lines 77–91) has no automated test — see §6.4 |
| 10 | **Labels from later names** | `attach_labels()` filling a missing ticker/company name from a *future* record (e.g. the acquirer's name after a merger, or a renamed ticker) | `portfolio.py::_asof_fill` uses `pd.merge_asof(..., direction='backward')` — only ever the most recent label dated `<= eom`, first from the raw panel, then from 8-K filing labels, else `'UNLABELED'` | `tests/test_portfolio.py::test_attach_labels_never_uses_future_dates` (synthetic, includes a deliberately-future label row that must be skipped); `tests/test_integrity.py::test_attach_labels_never_uses_future_label_real` (real data: independently re-derives the asof match, confirms it agrees with `attach_labels`' output, confirms the matched date is never after `eom`, and injects a `'FUTURE_SENTINEL'` label one month after a real holding's `eom` to confirm it never surfaces) |
| 11 | **Model-side look-ahead in FinBERT** | A frozen sentiment model that was itself trained (or fine-tuned) on data after the events it's scoring would not be "prediction," it would be "recall" (brief p.17) | `src/text.py`: `FINBERT_MODEL = 'ProsusAI/finbert'` pinned at `FINBERT_REVISION` (a specific commit hash), loaded with `use_safetensors=False` so a newer safetensors copy transformers might otherwise prefer can't silently override the pin. Module docstring documents that ProsusAI/finbert's training corpora (BERT-base: BooksCorpus+Wikipedia, 2018; further pretrained on Reuters TRC2, 2008–2010; fine-tuned on Financial PhraseBank, labelled 2014) all predate 2021, i.e. predate every 8-K this project scores (A7) | `build_text_features()` asserts every cached score row's `revision == FINBERT_REVISION`; `tests/test_text.py::test_score_finbert_chunks_resume_and_consolidate` checks the revision is recorded in output. **Note:** the training-corpus-date claim itself is a citation, not something code can assert — see §6.4 |
| 12 | **Market data / return misalignment** | Indexing the risk-free rate or S&P return by the *formation* month instead of the *holding* month (an off-by-one that quietly borrows next month's macro data, or silently NaNs a whole month) | `portfolio.backtest()` looks up `market.loc[holding_month]` where `holding_month = mth + MonthEnd(1)`, and raises `ValueError` (never a silent NaN) if that month is absent | `tests/test_portfolio.py::test_backtest_tiny_synthetic` (constructed so `market` is keyed *only* at holding months, each with a distinctive value that would never match if the code indexed by formation month instead), `test_backtest_raises_if_holding_month_missing_from_market` |
| 13 | **Multiple testing / result-driven decisions** | Brief p.12: *"an agent allowed to search freely will find spurious patterns faster than any human ever could. Multiple-testing discipline is your responsibility, not the model's."* Trying many specifications and reporting the best one after seeing test-period results | Pre-registration (`docs/research_log.md`, 2026-09-27 entry) fixed the headline strategy before any 2021–2026 result existed; every later amendment (A1–A14, §6.3) is timestamped against whether test-period results existed yet; the one amendment made *after* seeing a test-period table (A12, headline switch) is justified on **validation-window-only** evidence (a split-half pseudo-OOS test) and disclosed candidly rather than hidden; ablations (`pred_gate` etc.) are reported but the code comment in `MAIN.py` (step 6/8) states explicitly they are never promoted to headline post hoc | **No automated test** — this is a process control, not a code invariant. It is defended entirely by the timestamped disclosure trail in `docs/research_log.md` and `docs/SPEC.md` §10. Flagged as inherently process-only in §6.4 |

---

## 6.2 The integrity test suite

### 6.2.1 `tests/test_integrity.py`

This is the file a judge would run first — it is deliberately somewhat self-contained (per its own
module docstring) even though a few checks overlap with the per-module test files. **10 tests**:
9 marked `@pytest.mark.slow` (they read the real parquet files and/or fit real models), 1 fast
(`test_schedule_and_inner_holdout`, pure synthetic data).

| Test | What it checks | How |
|---|---|---|
| `test_target_alignment_real` | `stock_exret(t) == raw ret_exc(t+1)`; `target_month == eom + 1 month-end` on every row | Samples 5,000 labelled panel rows, re-derives the next-month raw `ret_exc` by merging on `(permno, target_month)`, requires match rate > 99.9% (checked on **every** row, not sampled, for the `target_month` identity) |
| `test_truncation_invariance_panel_real` | Panel features (ranks, `miss_` flags, `beta`, `size_z`, `gics2`) at month *T* use no information from after *T* | Rebuilds the panel in-memory from real data truncated at `eom <= 2021-06-30` (with that month's own `ret_exc_lead1m` wiped to NaN, since a real run wouldn't have it yet either) via `data._build()`, and checks month-*T* rows are numerically identical to the full build |
| `test_truncation_invariance_market_state_real` | Same invariance for `market_state()` (rolling market return/vol/dispersion) | Monkeypatches `data.load_chars` to truncate at *T*, rebuilds `market_state()`, compares row *T* |
| `test_truncation_invariance_text_real` | Same invariance for text features, on the **real** 8-K archive (the per-module `test_text.py` version uses only a tiny synthetic filings frame) | Truncates the real filings parquet at `filing_date <= T`, rebuilds `build_text_features()`, compares month-*T* rows |
| `test_feature_hygiene_real` | No banned column (`stock_exret`, `ret_exc_lead1m`, `target_month`, `has_filing`, `ticker`, `company_name`), no `*_raw` aux column, and no id/aux column (`gics2`, `beta`, `size_z`, `me`, `size_grp`, `date`, `eom`, `permno`) ever appears in the real feature set | Builds the real panel + text features, checks `feature_columns(panel) | TEXT_FEATURES` against the banned sets |
| `test_schedule_and_inner_holdout` (fast) | Per test-year: `max(train target) < min(valid target) < min(test target)`; union of all test months across years is exactly 2021-01..2026-08; the inner tuning holdout (last 12 target months of train, used by `_fit_specialist`) sits strictly inside train, at its tail | Synthetic 5-permno panel spanning 2015-01..2026-07 |
| `test_mini_end_to_end_pipeline` | The full `MAIN.py`-shaped pipeline (`run_all` → `smooth` → `backtest` → `attach_labels` → `write_submission` → `run_evaluation` → `calibrate` → `missing_return_sensitivity`) composes correctly, on a 600-permno real-data subset, test_year=2021 only | Checks holdings mechanics (100–500 names, legs sum to ±1, `MAX_WEIGHT` respected), the A10 filer-net constraint, `write_submission`'s exact ±100.000000 rounding, and that `evaluate.run_evaluation`/`calibrate`/`missing_return_sensitivity` all run without error on this subset. This is a *mechanics* smoke test — its docstring is explicit that no test-period performance is reported here, only interface correctness |
| `test_shuffled_label_ic_real` | Shuffling `stock_exret` within month across train+valid destroys any genuine label relationship, so the fitted `pred_ew` should show ~zero IC on the (real, unshuffled) test labels | See §6.2.2 below — this is the statistically careful version |
| `test_attach_labels_never_uses_future_label_real` | `attach_labels()` never uses a label dated after `eom`, on real data | See threat #10 above |
| `test_universe_invariance_to_nan_ret_exc_lead1m_real` | Universe membership is unchanged when `ret_exc_lead1m` is NaN'd for a random third of real rows | See threat #2 above |

### 6.2.2 The shuffled-label test, and why a single shuffled fit is not a valid test

Both `tests/test_integrity.py::test_shuffled_label_ic_real` and the lighter, fully-synthetic
`tests/test_models.py::test_shuffled_label_ic_small` exist because of a specific, documented
failure mode: **a single shuffled-label fit is not statistically trustworthy**, even though
shuffling within month destroys the true label relationship.

The reasoning, taken directly from the two tests' docstrings:

- Gradient boosting (LightGBM) can overfit its own inner-holdout noise onto one feature/direction
  by pure chance, landing a modest (~0.02–0.08) spurious IC against the *real* test-year labels —
  of either sign. This is a **per-fit event**, not sampling noise that shrinks with more stocks.
- `test_integrity.py`'s own docstring records that a prior draft of the test, run as a *single*
  6-specialist `run_all()` shuffle, gave **−0.0229 and +0.0383** on two separate single-rep runs —
  inconsistent in sign, both individually plausible as noise, and neither distinguishable from a
  genuine small leak at a fixed ±0.02 threshold.

The fix used here: treat the mean test-IC of `pred_ew` over **R independent reps** (each an
independently-seeded reshuffle, `R=6` in `test_integrity.py`, `R=16` in the lighter/cheaper
`test_models.py` version) as an i.i.d. sample of a mean-zero null, and test the *sample mean*
against a bound scaled by the *observed between-rep spread*:

```
bound = 2.5 * sd_over_reps / sqrt(R) + 0.005
assert abs(mean_ic_over_reps) < bound
```

This is a loose, two-sided ~99% t-style bound on the mean; the `+0.005` floor keeps the bound from
collapsing toward zero if the observed spread happens to be small by chance. `test_models.py`'s
lighter version instead pools `n_reps=16` single-specialist fits' monthly ICs directly and checks
`abs(pooled_mean_ic) < 0.02` — cheaper because it fits one specialist (not all six + gate) per
rep, so it can afford more reps for the same tightness.

### 6.2.3 Label-corruption invariance tests (the other half of "no leakage")

Truncation invariance proves features don't see the future; these tests prove **labels don't leak
into fitting/tuning they shouldn't touch**:

- `tests/test_models.py::test_specialist_ignores_valid_labels` — corrupting valid-window
  `stock_exret` with Gaussian(0, 50) noise, refitting a single specialist, and checking its
  test-set predictions are bit-identical (`atol=1e-6`) to the uncorrupted fit.
- `tests/test_models.py::test_run_all_ignores_valid_labels` — the same corruption at the full
  `run_all()` level: every specialist, `lgbm_all`, `pred_ols`, `pred_ew`, `pred_ew_notext` must be
  **exactly** unaffected; `pred_ridge`/`pred_lasso`/`pred_enet`/`pred_gate`/`pred_gate_notext` are
  *expected* to change (A2 explicitly allows linear-baseline alpha selection and the gate to use
  the official valid window) — the test asserts at least one of those does change, as a sanity
  check that the corruption actually did something.
- `tests/test_models.py::test_real_panel_ignores_test_labels` — the strongest of the three, on
  **real data**: corrupting test-split `stock_exret` with Gaussian(0, 50) noise and re-running
  `run_all()` must leave **every single column in `models.PRED_COLS`** (16 columns: every
  baseline, every specialist, both gates, `pred_ew_ret`, `pred_ew`, `pred_ew_notext`) exactly
  unchanged. This is the direct, real-data proof that nothing in the fitting/tuning path ever
  reads a test label.

### 6.2.4 Runtime and how to run

```
python -m pytest -q                 # everything (107 tests)
python -m pytest -q -m "not slow"   # 88 tests, no real-data reads — verified 2026-09-28: 58s, all passed
python -m pytest -q -m "slow"       # 19 tests: real parquet reads + real LightGBM fits
```

The `slow` marker (`pytest.ini`: `markers = slow: touches real data`) is applied per SPEC §0's
rule — *only* tests reading the real parquet files (or, for `test_shuffled_label_ic_real`/
`test_mini_end_to_end_pipeline`, real-data-derived fits) are marked slow; everything else runs on
synthetic data in seconds. The `slow` suite's cost is dominated by `test_mini_end_to_end_pipeline`
and `test_shuffled_label_ic_real` (each runs several `models.run_all()` calls — 6 independent
LightGBM+gate fits for the shuffled-label test alone) — their own docstrings put a single
`run_all()` call on the 600-permno fixture at roughly 25–30 seconds with the suite's patched
speed knobs (`_patch_mini_config`), so the shuffled-label test alone is on the order of 3 minutes;
this chapter did not re-time the full slow suite.

### 6.2.5 Other integrity-relevant tests, by module

Beyond `test_integrity.py`, these tests in the per-module files directly support the threat model
in §6.1 (not an exhaustive list of all tests in those files — see the individual chapters for the
non-integrity tests):

- **`tests/test_data.py`**: `test_universe_mask_ignores_ret_exc_lead1m`, `test_target_month_and_stock_exret`, `test_feature_columns_excludes_target_derived`, `test_market_state_truncation_invariance`, `test_miss_flags_use_only_cutoff_rows`, `test_real_panel_truncation_invariance`, `test_real_market_state_truncation_invariance`, `test_real_stock_exret_matches_next_month_raw`.
- **`tests/test_text.py`**: `test_truncation_invariance`, `test_build_text_features_month_assignment_and_counts`, `test_add_text_features_zero_fill_and_has_filing` (proves `has_filing` is *not* in `TEXT_FEATURES` but non-filers get exact 0 for every text feature), `test_filing_coverage_by_exit_buckets_and_censoring` + real `test_filing_coverage_by_exit_runs`, `test_build_text_features_raises_when_scores_settings_mismatch` (×2), `test_build_text_features_raises_when_scores_missing_document_ids`.
- **`tests/test_models.py`**: `test_char_groups_partition_all_147` (every characteristic is in exactly one group — no double counting, no omission), `test_train_excludes_null_label_valid_and_test_keep_it`, `test_make_target_ignores_null_label_rows`, `test_zscore_by_eom_zero_for_nan_rows`, `test_run_all_valid_includes_null_label_rows` (A11).
- **`tests/test_portfolio.py`**: `test_optimize_month_filer_net_neutral`, `test_optimize_month_forces_sector_relaxation` and `test_optimize_month_sector_lopsided_demeaning_fixes_feasibility` (regression tests for two real bugs found during development — see §6.3), `test_attach_labels_never_uses_future_dates`, `test_backtest_raises_if_holding_month_missing_from_market`, `test_write_submission_rejects_null_labels`.

---

## 6.3 Research-process integrity

### 6.3.1 Pre-registration

`docs/research_log.md`'s first entry (2026-09-27) fixes the headline strategy — *ridge gate over
5 specialists (4 characteristic-group LightGBMs + text LightGBM), interacted with `STATE_VARS`,
delivered via the neutral optimizer* — **before any 2021–2026 result is computed**, and states
plainly: *"Ablations are reported but never used to switch the headline."* (That last clause was
itself later amended — post-hoc and disclosed — by A12; see below.) It also records the
deviations already made from the strategy PDF at that point (no hold buffer, no FinBERT
embeddings/PCA, 2 gate state variables, no specialist refit on validation, portfolio penalties
calibrated on 2019–2020 validation only).

### 6.3.2 Amendments A1–A14

All fourteen amendments are logged in `docs/SPEC.md` §10 and `docs/research_log.md`, each dated
2026-09-27. "Pre-test" below means decided/logged before any 2021–2026 test-period number existed
(verified against the timeline entry in §6.3.4); "post-hoc" means decided after.

| # | What | Why | Timing |
|---|---|---|---|
| A1 | Drop `has_filing` as a model feature; text specialist trains/predicts on filer rows only; filer-only z-score (non-filers → exact 0) | 8-K filing coverage encodes future survival (measured, see §6.1 threat 8) | Pre-test |
| A2 | Specialists/`lgbm_all` tune on an inner holdout (last 12 months of train), refit on full train; the 24-month valid window reserved for the gate + linear-baseline alpha | Keeps specialist forecasts on valid genuinely out-of-sample | Pre-test |
| A3 | Portfolio penalty calibration uses `pred_ew` on 2019–2020 validation only, then locked; amended same day to calibrate on the **smoothed** `pred_ew` signal and reuse `backtest()`'s own optimizer call (so calibration enforces the A10 filer-net constraint exactly as the real backtest does) | Decided without seeing any test-period returns | Pre-test |
| A4 | Missing next-month returns (0.47% of test-universe stock-months, all true exits) stay 0 in the headline; an adverse −30%/+30% sensitivity is reported separately; never zero-filled for training | Delisting-return handling should be disclosed, not hidden or silently favorable | Pre-test (a data-handling rule, not a result-driven choice) |
| A5 | `returns.csv` uses first-of-holding-month `Date`; adds `total_ret_net`/`active_ret_net`; turnover stated as target-to-target; benchmark Sharpe reported N/A; exposure table reports min/avg/max ranges | Reporting-format hygiene | Pre-test |
| A6 | Single self-contained `MAIN.py` required at integration | Submission-format requirement | Pre-test |
| A7 | FinBERT revision pinned; deck cites pre-2021 training corpora | Model-side look-ahead guard (brief p.17) | Pre-test |
| A8 | Deck-visible holdings whose label isn't the same-month panel label get verified against SEC EDGAR by CIK, URL recorded in `label_audit.csv` | Label-quality assurance for the deck | Pre-test (**verification itself is a pending manual step** — see §6.4) |
| A9 | Five characteristic specialists (value, momentum, quality, investment_growth, risk_liquidity); `ebit_bev`/`sale_bev` move value→quality | Decided **on economic grounds**, matching the JKP-style taxonomy (FF5 RMW/CMA, HXZ ROE/I·A, Stambaugh-Yuan PERF/MGMT) | **Decided** pre-test, but **implemented** (models.py actually running 5 groups) after the smoke-run table existed — see §6.3.4 |
| A10 | Filer-net-neutral optimizer constraint: `\|sum of w over has_filing==1\|` ≤ `SECTOR_TOL` | Opus review finding: filer share of top-250 was 61% vs 48% base under `pred_ew` — the filer-only text z-score mechanically pushes filers into the signal tails | Pre-test |
| A11 | Valid-window predictions made for all rows including null-label future exits; gate fit on labelled rows only | Consistency of the valid-split schema with test | Pre-test |
| A12 | Headline switched from the ridge gate to the equal-weight blend (`pred_ew`) of the six specialists | See §6.3.3 | **Post-hoc, disclosed** |
| A13 | Specialist rounds/`num_leaves` chosen by mean monthly IC on the inner holdout, not pooled MSE (MSE early stopping was selecting 1-tree/7-distinct-value degenerate models); gate ridge alpha grid scaled by n | Found via validation diagnostics | Post-hoc, disclosed |
| A14 | Optimizer candidate selection sector-demeaned before ranking into long/short candidates | Real (and validation-window) formation months could be sector-lopsided enough that no relaxation step was feasible (worst-sector minimum net exposure 9.85% > 6% ladder ceiling in 3–14 of 24 validation months for some test years) | **Decided without seeing any test-period returns** — diagnosed purely from validation-window infeasibility |

### 6.3.3 A12 in more depth: why the headline is `pred_ew`, not the gate

The pre-registered headline was the ridge regime gate. It was demoted to an ablation because,
inside each 24-month **validation** window only, a split-half pseudo-out-of-sample test (fit on
one 12-month half, evaluate on the other) gave mean IC: gate −0.012, gate with n-scaled alpha
0.004, gate without state 0.009, NNLS 0.025, IC-weighted 0.038, **equal-weight 0.044** (best in 5
of 6 years, never negative). The mechanism cited (`docs/research_log.md`): roughly 24 monthly
observations cannot reliably identify the gate's 18 parameters; pooled-MSE fits flip factor signs
after a single bad factor year; state interactions add noise rather than signal — the same
diversification-beats-optimization logic as DeMiguel, Garlappi & Uppal (2009) on naive 1/N
weighting. The gate is retained in the codebase and reported as an ablation/explainability
exhibit (`MAIN.py` step 6/8 comment: *"this is a failed-on-validation ablation, not a lesser
alternative promoted post hoc from test results"*), never as the headline.

### 6.3.4 The smoke-run disclosure — full timeline, for candor

An interim `MAIN.py` run (old model code — 4 characteristic specialists, MSE-based early
stopping, no tone features), run before A9/A12/A13/A14 were implemented, ran far enough before
crashing in portfolio calibration that it wrote a complete test-period OOS R²/IC table to
`outputs/tables/oos_r2.csv`. That table was copied **verbatim, before any later run could
overwrite it**, to `docs/smoke_run_oos_r2_2026-09-27.csv` (reproduced below in full):

| model | mean_ic |
|---|---|
| pred_ridge | 0.059 |
| pred_lgbm_all | 0.056 |
| pred_spec_risk_liquidity | 0.070 |
| **pred_spec_text** | **−0.005** |
| pred_gate | 0.022 |
| **pred_ew** | **0.065** |
| pred_ols | 0.041 |
| pred_lasso | 0.046 |
| pred_enet | 0.046 |
| pred_spec_value | 0.034 |
| pred_spec_momentum | 0.045 |
| pred_spec_quality | 0.045 |
| pred_gate_notext | 0.021 |
| **pred_ew_notext** | **0.067** |

What this table was, and was not, used for:

- **A9** (fifth specialist, `investment_growth`) was *decided and logged on economic grounds*
  before this table existed, but its *implementation* in `models.py` happened after. The smoke
  run itself ran the pre-A9, 4-specialist code — it is not evidence for A9 either way.
- **A12** (gate → `pred_ew`) and **A13** (IC-based tuning) were both **decided after** this table
  was seen. Per `docs/research_log.md`, the evidence actually cited for each is
  validation-window-only (A12: the split-half pseudo-OOS test in §6.3.3; A13: the 1-tree
  degenerate-model failure mode found in validation diagnostics, independent of this table). This
  test-period table is disclosed "for candor" (the research log's own words) — it was not cited
  to justify either change, and the change each amendment made (equal-weight over gate; IC over
  MSE tuning) is not simply "whichever number in this table was higher" — e.g. this smoke table
  actually shows `pred_ew_notext` (0.067) beating `pred_ew` (0.065), yet the final headline
  *keeps* the text specialist. That decision does not rest on this table, or on any
  "use all available data sources" principle — `docs/SPEC.md` §0 states no such principle. It
  rests on the actual validation-only evidence: a full 24-month validation-window equal-weight IC
  comparison with vs. without text, by test year 2021..2026, gave
  0.016/0.069/0.030/0.054/0.052/0.045 (with text) vs. 0.015/0.070/0.032/0.053/0.055/0.047
  (without) — text-as-item-counts is neutral on validation, so it stays in the pre-registered
  6-specialist design; its FinBERT tone features (not yet scored as of this table) remain the
  untested part of that case.
- **A14** (sector-demeaned candidates) was decided from optimizer infeasibility in **validation**
  months only — no test-period returns involved at all.
- The A3 amendment (calibration on the smoothed signal, reusing `backtest()`'s own optimizer
  call) is also disclosed as concurrent with this timeline.

The honest reading for a judge: implementation timing for A9 lagged its decision, which is a
process imperfection worth naming rather than hiding, but the *content* of every post-smoke-run
decision (A12, A13) is traceable to validation-only evidence documented separately from the
smoke-run table, and the log states this explicitly rather than asserting it without a citation
trail.

### 6.3.5 Multi-agent build and adversarial verification

The codebase was built and reviewed through a multi-agent process referenced throughout
`docs/SPEC.md` and `docs/research_log.md` as "Opus review" / "Opus finding" / "Opus audit" —
i.e., builder agents wrote the modules against the shared spec, and a separate verification pass
(rather than the builder grading its own work) adversarially reviewed it. Bugs it is credited
with catching, each traceable to a specific amendment or regression test:

- **A1 survivorship leak**: the original spec would have let `has_filing` (and an unfiltered text
  specialist) leak future-survival information; caught before implementation.
- **A10 filer-tail concentration**: found that the filer-only text z-score mechanically pushed
  filers into the signal tails (61% of top-250 vs 48% base), which the pre-A10 spec did not
  constrain.
- **A14 sector-lopsided candidates**: found that real/validation formation months could be
  sector-lopsided enough that the optimizer's relaxation ladder could not reach feasibility
  (worst-sector net exposure 9.85% vs a 6% ceiling) — root-caused to candidate *selection*, not
  the solver, and fixed by sector-demeaning before candidate ranking.
- **A13 degenerate LightGBM models**: MSE-based early stopping was found to select 1-tree models
  with as few as 7 distinct predicted values — caught via validation diagnostics, not a code
  review of the training loop alone, and fixed by switching specialist model selection to IC on
  an inner holdout (regression-tested by `tests/test_models.py::test_specialist_not_degenerate`,
  which asserts >50 distinct `pred_spec_quality` values per test month).
- **`_check_constraints` relaxed-tolerance bug** (`docs/research_log.md`, "2026-09-27 —
  portfolio.py" entry): `_check_constraints` was asserting every exposure group against the
  *fixed* config tolerances even on months where `solve_ladder` had only found a feasible
  solution at a *relaxed* step, so the post-solve dust/rescale check fired a false-positive
  `AssertionError` on every relaxed month. Fixed by threading the actually-used
  `(sector_tol, size_tol, beta_tol)` through to the check; regression-tested by
  `tests/test_portfolio.py::test_optimize_month_forces_sector_relaxation`, which the log records
  was confirmed to reproduce the exact real-data failure mode ("sector A exposure breached" /
  real-world "sector 10 exposure breached") by temporarily reverting the fix.

Why this matters to the committee: the brief (p.12) explicitly asks teams building agentic
pipelines to name "the reproducibility agent" and to be honest about what it caught, treating a
candid account of failure as more valuable than a polished account of success. This project's
answer is the paper trail above — every bug an adversarial pass found is tied to a named
amendment, a research-log entry, and (where the bug was in code rather than in the spec) a
regression test that reproduces the original failure when the fix is reverted.

---

## 6.4 Known limitations and residual risks

Stated plainly, without spin:

- **Filing-coverage survivorship measurement has its own censoring bound.**
  `text.filing_coverage_by_exit()` classifies a permno as a confirmed "exit" only if its last
  appearance in the *raw* (unfiltered) characteristics data is 12+ months before the raw data's
  own last `eom`; permnos last seen 1–11 months before the end are excluded entirely as
  ambiguous/right-censored (`tests/test_text.py::test_filing_coverage_by_exit_buckets_and_censoring`
  documents this explicitly). This is the right conservative choice, but it means the 1.3%→59.7%
  gradient is measured on a subset of exits, not all of them.

- **Delisting/missing returns are set to 0 in the headline.** 0.47% of test-universe stock-months
  have no next-month return (true exits); the headline treats their contribution as 0 (A4), and
  labels are never zero-filled for training. An adverse sensitivity check
  (`portfolio.missing_return_sensitivity`, −30% for missing longs / +30% for missing shorts) is
  reported alongside, but is not blended into the headline number.

- **2026 characteristics/returns are sourced from Compustat, not CRSP.** Per the supplied data's
  own `readme.md` (§4, `source_crsp` column): *"All 2026 rows use Compustat."* Every prior year in
  this panel is CRSP-sourced. The pipeline does not — and cannot, from inside this repo — verify
  that Compustat's 2026 return computation is methodologically identical to CRSP's for the same
  stock-months; this is a data-vintage characteristic of the supplied panel, not a pipeline bug,
  and is disclosed here rather than left implicit.

- **Sector/size/beta tolerance relaxation is real, tracked, and sometimes needed.** A14 reduced
  how often the optimizer needs `RELAX_STEPS` beyond the base tolerance, but did not eliminate it;
  `evaluate.exposure_table()` reports `relax_share` and `relax_count_<step>` precisely so this is
  visible in the deck rather than hidden by an assertion that never fires. Measured post-A14: a
  feasibility sweep over all 92 real formation months (24 validation + 68 test), using a
  real-shaped mechanics signal, solved every month at base tolerances — 0 relaxations, 0 failures.

- **Turnover/cost are target-to-target, not intra-month-drift-aware.** `compute_month_return`'s
  own docstring is explicit: turnover is `0.5*sum|w_t - w_prev|` against the *previous target*,
  not weights drifted by intra-month price moves. A5 states this convention explicitly in the
  deliverable rather than silently reporting a lower-than-realistic turnover number.

- **Borrow costs are not modeled.** `config.COST_BPS = 10` (a flat 10bps per dollar traded, charged
  as `COST_BPS/1e4 × Σ|w_t − w_{t−1}|` — every buy and every sell counted once, not a one-way rate)
  is the only transaction-cost assumption anywhere in
  `compute_month_return`; there is no separate stock-loan/hard-to-borrow fee, no locate
  constraint, and no scaling of that cost by name (e.g. by the nano/micro-cap or
  low-dollar-volume concentration `evaluate.short_book_table()` reports for the short book). For
  a market-neutral book whose short leg skews toward smaller, less liquid names, real borrow
  costs could plausibly exceed this flat assumption — this is disclosed here as a residual risk on
  the net-of-cost numbers, not something the code currently models or reports a sensitivity for.

- **The gate (pred_gate) failed its own validation test and is not the headline.** This is
  disclosed at length in §6.3.3 — flagged here again because it is a genuine model limitation
  (the more sophisticated regime-conditioned combiner underperforms the naive equal-weight blend
  on held-out validation evidence), not a footnote.

- **The text specialist's standalone contribution is weak on the one test-period data point that
  exists, but validation-window evidence is neutral, not negative.** In the disclosed (never
  decision-driving) smoke-run table, `pred_spec_text` alone had mean IC **−0.005** (essentially
  indistinguishable from noise, one of only two negative entries in the whole table), and
  `pred_ew_notext` (0.067) slightly *beat* `pred_ew` (0.065). The actual case for keeping the text
  specialist is validation-only: a full 24-month validation-window equal-weight IC comparison with
  vs. without text, by test year 2021..2026, gave 0.016/0.069/0.030/0.054/0.052/0.045 (with text)
  vs. 0.015/0.070/0.032/0.053/0.055/0.047 (without) — from the orchestrator's validation-window
  gate diagnosis (2026-09-27), measured before FinBERT tone features existed, so this reflects the
  item-count text features only. Text-as-item-counts is neutral on validation, which is why it
  stays in the pre-registered 6-specialist design; its FinBERT tone features are not yet scored and
  remain the untested part of the case. (There is no "use all available data sources" principle in
  `docs/SPEC.md` §0; that framing was inaccurate and is corrected here.) These validation-window
  numbers are being logged in `docs/research_log.md` by the agent maintaining that log. This is
  worth being candid about if a judge asks "why keep a specialist that showed slightly negative
  test-period IC."

- **Labels for names requiring EDGAR verification (A8) are a pending manual step.** A8 commits to
  verifying, by CIK, any deck-visible holding whose `label_source` isn't the same-month panel
  label, recording the verification URL in `label_audit.csv`. This is a deck-production step, not
  something any test in `tests/test_portfolio.py` enforces — `write_submission()` only asserts
  `ticker`/`company_name` are non-null (`UNLABELED` satisfies that), not that the EDGAR check
  actually happened. Treat this as an outstanding to-do at deck time, not a completed control.

- **Filing timestamp precision.** The supplied data's `readme.md` states every filing has
  `filing_time_precision = 'date_or_midnight_placeholder'` — filing timestamps do not establish
  the exact intraday time a disclosure became public. Because this pipeline aligns everything at
  monthly granularity (`eom = filing_date + MonthEnd(0)`), intraday timing imprecision cannot by
  itself create look-ahead here, but a `filing_date` that were *itself* wrong by enough to cross a
  month boundary (a data-provider quality issue, not a pipeline bug) would not be caught by any
  test in this repo.

- **The `MAIN.py --reuse-preds` staleness guard is untested.** `MAIN.py` lines 77–91 check that a
  cached `preds.parquet` has every column `models.PRED_COLS` expects and is newer than
  `src/models.py`'s mtime before allowing `--reuse-preds` to skip refitting — a real (if narrow)
  leakage-adjacent risk (running an old model's cached predictions against new evaluation code)
  that the guard defends against but no test in `tests/` exercises either `RuntimeError` path.

---

## 6.5 FAQ: how to answer a judge

**Q1: "How do I know `stock_exret` isn't leaking into your features?"**
`stock_exret` is set exactly once (`data.py::_build`, from `ret_exc_lead1m`, never re-shifted) and
is one of three names hard-banned from every feature list by `models._assert_no_leakage()`, which
runs on every `run_all()` call. `tests/test_integrity.py::test_feature_hygiene_real` proves this
on the real, current feature set; `test_target_alignment_real` proves the label itself is
correctly aligned (>99.9% match against the raw next-month return, on a 5,000-row real sample).

**Q2: "Is your investable universe secretly conditioned on which stocks survived or had good
returns?"** No — `data.universe_mask()` uses only price and market-equity percentile; it never
reads `ret_exc_lead1m`. Proven on real data by NaN-ing a third of `ret_exc_lead1m` and showing the
universe mask is bit-identical (`test_universe_invariance_to_nan_ret_exc_lead1m_real`).

**Q3: "Could any of your preprocessing — ranks, missing flags, market state, text features — have
used information from after the formation date?"** No; each is checked by rebuilding it from data
truncated at a fixed date `T` and comparing month-*T* output to the full build — on **real data**,
for the panel, market state, and text features separately (`test_truncation_invariance_panel_real`,
`_market_state_real`, `_text_real`).

**Q4: "Did you tune anything — even hyperparameters — on the test set?"** No. This is the
strongest single proof in the suite:
`tests/test_models.py::test_real_panel_ignores_test_labels` corrupts real test-split labels and
re-runs `models.run_all` on a 400-permno real-panel subset for one test year (2021); **every one
of the 16 columns in `models.PRED_COLS`** (every baseline, every specialist, both gates, both
equal-weight blends) comes back bit-identical.
Nothing in the fitting/tuning path ever reads a test label — `r2_table()`, which does, is only
ever called afterward on a separate object.

**Q5: "You built a fairly sophisticated ridge gate but your headline is a plain equal-weight
average. Isn't that giving up on your own model?"** The gate underperformed on validation-window
evidence, not test — a split-half pseudo-out-of-sample test inside each 24-month validation window
gave the gate a mean IC around −0.01 to +0.01 depending on variant, versus +0.044 for equal-weight
(best in 5 of 6 years, never negative). With ~24 monthly observations and 18 gate parameters, this
is the expected failure mode of an over-parameterized combiner, not a coincidence — the mechanism
mirrors DeMiguel/Garlappi/Uppal (2009) on naive 1/N. The gate is kept as an ablation and
explainability exhibit, never as the headline.

**Q6: "You disclosed that a smoke run showed you test-period results before finalizing some
amendments. Isn't that leakage?"** We disclose it precisely so it can be checked rather than
found. The smoke run (old, pre-A9 4-specialist code) crashed mid-run but wrote a full OOS R²/IC
table, which was copied verbatim to `docs/smoke_run_oos_r2_2026-09-27.csv` before any later run
could overwrite it — full timeline in §6.3.4. Two amendments (A12, A13) were decided after this
table existed; both are justified in `docs/research_log.md` with validation-window-only evidence
(a split-half pseudo-OOS test for A12; a degenerate-model diagnosis for A13), not by this table —
and notably, this table doesn't even support the decision that was made in one visible case (it
shows the no-text blend beating the with-text blend, yet the final pipeline kept text). A14 (the
other post-A9 amendment) used purely validation-window infeasibility evidence with zero test-period
involvement. The one process gap we name ourselves: A9 was decided pre-table but *implemented*
post-table — a timing imperfection, disclosed rather than smoothed over.

**Q7: "How do you handle survivorship bias in your 8-K text data?"** We measured it directly:
same-month 8-K filing coverage is 1.3% for stock-months at their last panel appearance, rising to
59.7% for stocks that survive to the panel's end (`text.filing_coverage_by_exit()`). Because that
gradient would leak future survival if used as a feature, `has_filing` is excluded from the model
feature set entirely; the text specialist is trained and predicted on filer rows only, its
within-month z-score gives non-filers an exact 0 (not an imputed/interpolated value); and the
optimizer enforces a hard filer-net-neutral constraint so the resulting filer-skewed signal can't
mechanically tilt the book.

**Q8: "Could FinBERT itself have seen the future — i.e., been trained on data after the 8-Ks it's
scoring?"** No: it's pinned to a specific commit revision (`FINBERT_REVISION` in `src/text.py`,
loaded with `use_safetensors=False` to block a newer weights copy from silently overriding the
pin), and its documented training corpora — BERT-base pretraining on BooksCorpus+Wikipedia (2018),
further pretraining on Reuters TRC2 newswire (2008–2010), fine-tuning on the Financial PhraseBank
(labelled 2014) — all predate 2021, which predates every 8-K in this dataset.

**Q9: "What happens to stocks that delist mid-test-period — do you just make the problem
disappear?"** No: 0.47% of test-universe stock-months have no realized next-month return (A4);
the headline treats their return contribution as 0 (disclosed, not hidden), and a separate adverse
sensitivity report (−30% on missing longs, +30% on missing shorts) is shown alongside it via
`portfolio.missing_return_sensitivity`. Labels are never zero-filled for *training* — a missing
label means the row is simply excluded from the training set (`splits()` requires a non-null
label for train).

**Q10: "Are short-borrow costs reflected in your Sharpe/IR?"** No — only a flat 10bps
transaction cost (`config.COST_BPS`, charged per dollar traded on every buy and every sell) is
modeled, applied uniformly regardless of how hard a name is to borrow. Given the short book skews toward smaller/less-liquid names (see
`evaluate.short_book_table()`), this is a genuine limitation on the net-of-cost numbers, disclosed
in §6.4 rather than glossed over.

**Q11: "How do you know a holding's ticker/company name in your submission isn't the *later*,
post-event name (e.g. after a merger or rename)?"** `attach_labels()` only ever fills a missing
label from the most recent record dated on or before that holding's formation month
(`pd.merge_asof(..., direction='backward')`), checked first against the raw panel, then 8-K filing
labels, else `'UNLABELED'`. On real data, this was independently re-derived and cross-checked
(`test_attach_labels_never_uses_future_label_real`), including injecting a synthetic
`'FUTURE_SENTINEL'` label one month after a real holding's formation date and confirming it never
surfaces.
