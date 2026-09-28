# Research log

## 2026-09-27
Pre-registration: headline strategy = ridge gate over 5 specialists (4 characteristic-group LightGBMs + text LightGBM), interacted with STATE_VARS, delivered via the neutral optimizer. Fixed before any 2021-2026 result is computed. Ablations are reported but never used to switch the headline. Deviations from the strategy PDF: no hold buffer (EMA + turnover penalty suffice), no FinBERT embeddings/PCA (tone + item codes only), gate uses 2 state variables, specialists fitted on training window only (no refit on validation), portfolio penalties calibrated on 2019-2020 validation predictions only then locked.

## 2026-09-27 (after Opus spec review)
Amendments locked in docs/SPEC.md section 10, in response to an Opus review of the pre-registered spec:
A1 Survivorship: 8-K filing coverage encodes future survival (permnos exiting within 12 months have a same-month filing 0-2.6% of the time vs 40-58% for survivors), so `has_filing` is dropped as a feature; text.TEXT_FEATURES is within-filer features only, the text specialist trains/predicts on filer rows only, and its within-month z-score is computed among filers with non-filers set to exactly 0. Disclosed in the deck.
A2 Validation hygiene: specialists and lgbm_all tune num_leaves/early stopping on an inner holdout = last 12 target months of train, then refit on the full training window with the chosen settings. The 24-month validation window is reserved for the gate and for linear-baseline alpha selection (template baselines only). Gate ridge alpha chosen by GroupKFold CV grouped on eom.
A3 Portfolio penalty calibration uses pred_ew (no fitted combiner) on the 2019-2020 validation predictions, then locked.
A4 Missing next-month returns (0.47% of test-universe stock-months, all true exits) stay 0 in the headline; an adverse sensitivity (-30% longs / +30% shorts) is reported separately. Labels are never zero-filled for training.
A5 returns.csv: first-of-holding-month Date (matches holdings.csv), adds total_ret_net and active_ret_net; total_ret (gross of costs) remains the headline. Turnover is target-to-target, stated as such. Benchmark Sharpe reported N/A. Exposure table reports min/avg/max ranges.
A6 Submission requires a single MAIN.py: final integration step bundles all src modules into one self-contained MAIN.py.
A7 FinBERT revision pinned (hash recorded in src/text.py FINBERT_REVISION); deck cites its pre-2021 training corpora (BERT 2018, TRC2 2008-10, Financial PhraseBank 2014).
A8 Deck-visible holdings whose label_source is not the same-month panel label are verified against SEC EDGAR (by CIK); the verification URL is recorded in label_audit.csv.

## 2026-09-27 — Amendment A9 (decided on economic grounds before any 2021-2026 result was seen)
Characteristic specialists become five: value, momentum, quality (profitability/earnings quality/safety), investment_growth (asset/investment growth, accruals, issuance: FF5 RMW vs CMA, HXZ ROE vs I/A, Stambaugh-Yuan PERF vs MGMT), risk_liquidity. ebit_bev and sale_bev move from value to quality (profitability/turnover on book EV, not price ratios). Headline = gate over 6 specialists (5 char + text).

## 2026-09-27 — FinBERT scoring moved to GPU (DGX Spark); decided before any FinBERT result was seen
The CPU-only scoring run (373,139 8-K filings, ~5-6 docs/s on this laptop, ~20h ETA) was compute-bound at MAX_LENGTH=128 (an "event lead": clean_text() drops the cover page and boilerplate item title, then the tokenizer truncated to the first 128 tokens, since a full-document pass didn't clear a same-day runtime -- see the benchmark comment above MAX_LENGTH in src/text.py). The user has an NVIDIA DGX Spark (GB10 Grace Blackwell, CUDA GPU, 128GB unified memory); on that device compute is no longer binding. Decision, made before any GPU score was computed: MAX_LENGTH moves to 512 (BERT's own limit -- the full cleaned event body per filing, not just its opening sentences), precision moves to fp16 (checked against fp32 via `--check` before the full run; fp32 stays the CPU default), and batch size moves to 256. Runbook: docs/RUN_FINBERT_DGX.md. The original CPU run (ml=128, fp32) is kept only as a fallback and is not blended with or overwritten by the GPU run -- outputs are settings-specific filenames (`finbert_scores_L{max_length}.parquet`) so the two never mix.

Same-day addendum (Opus review of src/text.py, before any FinBERT score was seen): clean_text() also gained (1) sentence-level stripping of recurring safe-harbor/furnished/incorporated-by-reference boilerplate (measured on a 4% filing sample: Item 2.02 windows, 33% of the corpus, were >=2-phrase boilerplate 82.5% of the time; 7.01 55%; 8.01 30%) and a hard cut of the Item 9.01/exhibits/signature tail; (2) tighter name/ticker masking (short suffix-stripped name variants now case-sensitive and gated on length/word-count; tickers masked only in an explicit exchange-context, e.g. "NYSE: TICK", to stop bare-ticker false positives on ordinary words); (3) an empty-cleaned-text filing now scores NaN (score_texts_safe) rather than being scored as an empty string, excluded from tone aggregates. These are cleaning-precision fixes, not signal-motivated, and were made before any 2021-2026 backtest result existed.

DGX run postponed by the user; before the full run, a local CPU smoke test (`--limit`, writes to outputs/cache/finbert_smoke/, never the real chunk dir or finbert_scores_L512.parquet) was used to sanity-check the new cleaning/masking/scoring path end to end -- see the same day's SubagentHandback report for example (item, cleaned-lead, pos/neg/neu) rows and throughput observed.

## 2026-09-27 — A12/A13 (models.py: headline switch and specialist tuning)
A12 (post-hoc, disclosed): headline signal switched from the ridge regime gate to the equal-weight blend of the six specialist z-scores (pred_ew). Disclosure: an interface smoke run had printed test-period ICs (gate 0.022 vs equal-weight 0.065) before this decision. The decision rests on validation-window evidence only: a split-half pseudo-out-of-sample test inside each 24-month validation window (fit on one 12-month half, evaluate on the other) gave mean IC gate -0.012, gate with n-scaled alpha 0.004, gate without state 0.009, NNLS 0.025, IC-weighted 0.038, equal-weight 0.044 (best in 5 of 6 years, never negative). Mechanism: ~24 monthly observations cannot identify 18 gate parameters; pooled-MSE fits flip factor signs after a bad factor year; state interactions add noise (cf. DeMiguel, Garlappi & Uppal 2009 on 1/N). The gate is kept as an ablation and explainability exhibit.
A13: specialist boosting rounds and num_leaves are chosen by mean monthly Spearman IC on the inner holdout (last 12 target months of training), not pooled MSE (MSE early stopping selected 1-tree models, e.g. 7 distinct predictions). Rounds evaluated on a fixed grid, floor 100. Gate ridge alpha grid scaled by n (len(y) * logspace(-3, 2, 11)).

## 2026-09-27 — portfolio.py: _check_constraints relaxed-tolerance bug, and a real-data infeasibility diagnosis
Smoke run of MAIN.py (old model code: 4 char specialists, pre-fix) printed test-period OOS R2/IC before crashing in portfolio calibration. Observed: gate mean IC 0.022 vs equal-weight 0.065 and char specialists 0.034-0.070. Disclosed here for candor; the headline is not switched on this basis. The gate is being diagnosed for bugs using validation-window evidence only; any resulting change will be recorded as post-hoc.

Bug fixed (src/portfolio.py `_check_constraints`): it asserted every exposure group against the fixed config.SECTOR_TOL/SIZE_TOL/BETA_TOL even when solve_ladder had only found a feasible solution at a relaxed RELAX_STEPS step (e.g. "sector x2"), so the post-solve dust/rescale check fired an AssertionError on every relaxed month regardless of whether the relaxed tolerance was actually satisfied. Fixed by threading the effective (sector_tol, size_tol, beta_tol) actually used for the returned solve through to _check_constraints; the competition-fixed rules (long/short leg sum to +-1, MAX_WEIGHT cap, 100..500 names) stay checked against fixed constants since they are never on the relaxation ladder. Added a regression test (test_optimize_month_forces_sector_relaxation) with a synthetic month engineered so sector A's net exposure is forced (by MAX_WEIGHT capacity on the non-A candidates, independent of the signal) to lie strictly above SECTOR_TOL but at/under 2x SECTOR_TOL -- infeasible at the base tolerance for any weight choice, feasible only after 'sector x2'. Confirmed the test reproduces the exact real-data failure mode by temporarily reverting the fix: it fails with "sector A exposure breached", matching the observed real-data "sector 10 exposure breached".

Diagnosis of why real data needs relaxation (as requested, no design change made): loaded outputs/cache/panel.parquet, derived has_filing via src.text.add_text_features (no consolidated FinBERT score cache exists yet, so tone columns are omitted, but has_filing/n_filings come straight off the raw 8-K filings table and are unaffected), and ran optimize_month at three real formation months (2019-06-30, 2022-06-30, 2025-06-30) with a mechanics-only z-scored ret_12_1 signal, w_prev empty each time. At the BASE tolerances, all three months are feasible (relax='' in each case) using this simple momentum signal, but every one of them sits with the worst GICS2 sector's net exposure at exactly +-0.03000 -- i.e. the SECTOR_TOL=0.03 constraint is binding at its exact boundary in all three sampled months. In 2022-06-30 the filer-net constraint (A10) is SIMULTANEOUSLY binding at exactly -0.03000 alongside sector 35 also at exactly -0.03000; in 2025-06-30 filer-net binds at +0.03000 while the worst sector binds at -0.03000. Candidate-set composition explains the mechanism: with N_CAND=250 sign-fixed candidates per side, GICS2 sector 35 alone made up 45.2% of the short-candidate set in 2022-06-30 (vs 10.8% of long candidates), and filer share of short candidates was 66.4% vs 43.2% of long candidates (vs 44.2% base universe) that month -- a highly lopsided candidate mix that leaves the +-0.03 sector/filer bands with essentially zero slack once MAX_WEIGHT=0.015 caps how much any one leg's non-dominant-sector names can absorb. Because the true optimum sits exactly on these boundaries rather than comfortably inside them, the post-solve dust-zeroing + leg-rescale step (which necessarily perturbs each leg's sum back to exactly +-1 after dropping sub-1e-5 weights) has no room to stay within tolerance, so the fixed_zero re-solve frequently needs the 'sector x2' (or filer, which shares the sector ladder step) rung to find a feasible point again. This is consistent with A10's own finding (filer share of top-250 61% vs 48% base under pred_ew) that filer status is not spread evenly across the signal-ranked candidate set. No design change is implemented per instructions; a candidate-set change that draws long/short names more evenly per sector (or a larger N_CAND with the 500-name cap still enforced by the existing post-solve count assert) would give the tightest groups more names to net against and merits consideration outside this task's scope.

## 2026-09-27 -- Disclosure & timeline
The interface smoke run described above (old model code: 4 characteristic specialists, MSE-based
early stopping, no tone) ran far enough before crashing in portfolio calibration to write the full
test-period OOS R2/IC table to outputs/tables/oos_r2.csv. That table has been copied verbatim,
before any later run could overwrite it, to docs/smoke_run_oos_r2_2026-09-27.csv. Rows (mean IC):
pred_ridge 0.059, pred_lgbm_all 0.056, pred_spec_risk_liquidity 0.070, pred_spec_text -0.005,
pred_gate 0.022, pred_ew 0.065 (full table also has: pred_ols 0.041, pred_lasso 0.046,
pred_enet 0.046, pred_spec_value 0.034, pred_spec_momentum 0.045, pred_spec_quality 0.045,
pred_gate_notext 0.021, pred_ew_notext 0.067).

Timeline, for candor:
- A9 (fifth specialist, investment_growth) was DECIDED and logged ("on economic grounds... before
  any 2021-2026 result was seen") before this table existed, but IMPLEMENTED (models.py actually
  running five characteristic-group specialists instead of four) after it existed. The smoke run
  itself used the pre-A9, 4-specialist code.
- A12 (headline switched pred_gate -> pred_ew) and A13 (IC-based specialist round/num_leaves
  selection, motivated by the 1-tree bug found in validation diagnostics) were both decided AFTER
  this table was seen and are marked post-hoc in docs/SPEC.md section 10. The evidence actually
  cited for each is validation-window only (A12: split-half pseudo-OOS IC inside each 24-month
  validation window; A13: the MSE-early-stopping 1-tree failure mode found in validation
  diagnostics). This test-period table is disclosed here for candor; it was not used to justify
  either change.
- A14 (sector-demeaned candidate selection, src/portfolio.py) was decided because the optimizer
  was infeasible in 3-14 of 24 VALIDATION months (min achievable sector net exposure 9.85%,
  against the 6% cap reached after the 'sector x2' relaxation rung) -- decided without any
  test-period returns.
- A3 is amended: portfolio penalty calibration now runs on the SMOOTHED pred_ew 2019-2020
  validation signal (portfolio.smooth, not the raw within-eom z-scored pred_ew), and calibrate()
  now enforces the same filer-net constraint (A10) as the real backtest, since calibrate() was
  changed to reuse backtest()'s per-month optimizer call directly rather than a standalone loop.

See docs/SPEC.md section 10 (A14, A3 amendment) for the corresponding spec changes.

## 2026-09-27 -- int8 quantization benchmarked and rejected for FinBERT (logged retroactively 2026-09-28)
During the original 128-token CPU run, torch dynamic int8 quantization (`quantize_dynamic` on
`nn.Linear`) was benchmarked on CPU and rejected as a throughput option: the text agent measured
`corr(pos-neg)` int8 vs. fp32 = 0.686 on 200 docs. An Opus verifier re-checked this on 80 filings
at `max_length=128` (2 threads): int8 was 1.72x faster (37.4s vs. 64.4s) but tone correlation was
only 0.83 (Spearman 0.815), negative-class correlation 0.61, top-label agreement 98.75%, and mean
`|tone|` gap 0.066 against a tone s.d. of 0.131 -- int8 was rejected on precision grounds. fp16 on
GPU (DGX Spark) replaced it as the throughput solution once GPU access became available (see the
"FinBERT scoring moved to GPU" entry above). This history was not previously recorded in this log;
it is logged here for completeness, dated to the day it happened (2026-09-27).

## 2026-09-27 -- A1 correction: survivorship filing-coverage numbers
The A1 figure originally recorded ("0-2.6% vs 40-58%") was an early, less careful pass. Re-measured
with text.filing_coverage_by_exit() (exit = a permno's last appearance in the RAW characteristics
file, not the universe-filtered panel; a permno's last raw eom counts as a true exit only when it
falls 12+ months before the raw data's own last eom, so near-term/right-censored dropouts that could
still reappear past the data's edge are excluded entirely rather than misclassified; coverage is
reported over universe stock-months only, the rows has_filing/TEXT_FEATURES actually apply to).
Corrected same-month 8-K filing coverage, bucketed by distance from exit: last panel row 1.3%,
1-2 months before exit 3.3%, 3-5 months before exit 3.6%, 6-11 months before exit 3.9%, 12+ months
before exit 8.9%, survivors to panel end 59.7%. The qualitative conclusion (has_filing encodes
future survival and must not be a feature; A1's TEXT_FEATURES/A9/A10 design is unaffected) is
unchanged -- this is a numbers-only correction to docs/SPEC.md section 10 A1.

## 2026-09-27 -- text specialist on validation (logged retroactively 2026-09-28)
From the validation-window-only gate diagnosis (Opus, 2026-09-27; item-count text features only,
FinBERT tone not yet scored), the full 24-month validation-window mean IC of the equal-weight
blend WITH text vs. WITHOUT text, by test year 2021..2026, was 0.016/0.069/0.030/0.054/0.052/0.045
vs. 0.015/0.070/0.032/0.053/0.055/0.047 -- text-as-item-counts is neutral on validation. The text
specialist stays in the pre-registered six-specialist design; its FinBERT tone features are the
untested part.

## 2026-09-28 -- A15 (post-hoc, disclosed): beta model
Test-period realized beta was -0.33 (NW SE 0.16) and was seen before this decision. The fix rests
on validation-only evidence: the 2019-2020 validation backtest showed realized beta -0.43 (NW SE
0.19); cause: the optimizer piled into shorts with missing beta_60m imputed as 1.0 (46% of short
candidates, 70% of short weight; pre-2019 forward beta of such names 1.25; high-ivol shorts
understated). New estimator (formula), coefficients fitted on 2015-2017 formation months only;
N_CAND 250->350 because with 250 the short candidates were too beta-lopsided (validation beta
-0.23 vs -0.07 at 350); cardinality guard keeps <=500 names. Validation: beta -0.07 (NW SE 0.14),
2019 +0.03 / 2020 -0.10. Estimators rejected on validation: Dimson 21d blends (-0.43), missing->1.25
only (-0.32), FP blend B1 (-0.24/-0.16), vol-neutral constraint (infeasible at 250; large IR cost).
