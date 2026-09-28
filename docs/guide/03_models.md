# Chapter 3 — Training: the prediction problem, the schedule, the models

This chapter is about `src/models.py` (the file that turns a panel of ranked characteristics
and text features into monthly return forecasts) plus the pieces of `src/config.py`,
`src/data.py` and `src/text.py` that feed it. Every claim below is checked against the code as
of this writing; function references are `src/models.py:function_name` unless stated otherwise.
Two numeric tables in this chapter are explicitly disclosed-only, not results: the interface
smoke-run OOS R2/IC table (`docs/smoke_run_oos_r2_2026-09-27.csv`) and the validation-window
pseudo-OOS diagnosis behind Amendment A12 (`docs/research_log.md`, `docs/SPEC.md` §10 A12). The
full 2021-2026 training run has not been executed; nothing in this chapter should be read as a
final performance claim.

---

## 1. The prediction problem

The brief (p.13) poses the forecasting problem in the most general form used in the empirical
asset-pricing ML literature (Gu, Kelly & Xiu 2020; the brief also cites Goyenko & Zhang 2022):

$$r_{i,t+1} = E(r_{i,t+1}) + \epsilon_{i,t+1}, \qquad E(r_{i,t+1}) = g^*(z_{i,t})$$

Stock $i$'s excess return realized during month $t+1$ is decomposed into a conditional
expectation, produced by a fixed function $g^*(\cdot)$ of information available at the end of
month $t$, plus idiosyncratic noise $\epsilon_{i,t+1}$. $z_{i,t}$ is the $P$-dimensional vector
of predictors — here $P = 163$ (147 characteristics + 16 missingness flags, §3 below) — known as
of month-end $t$. $g^*$ is shared across all stocks and months (it "maintains the same form over
time and across different assets" — brief p.13); AlphaBERT does not fit a separate model per
stock.

In the repo's own column names: `eom` is the formation month $t$; `target_month = eom + 1`
month-end is $t+1$; `stock_exret` is `ret_exc_lead1m`, the already-led excess return realized
during `target_month` — i.e. `stock_exret` **is** $r_{i,t+1}$, indexed by the row's `eom`. This
is asserted at the row level, not just described: `_assert_no_leakage`
(`src/models.py:_assert_no_leakage`) forbids `stock_exret`, `target_month` and
`ret_exc_lead1m` from ever appearing among the 163 feature columns or the text feature columns,
and `tests/test_integrity.py::test_feature_hygiene_real` checks the same on the real panel.

### Target construction — `make_target`

```python
def make_target(df):
    y = df["stock_exret"]
    tm = df["target_month"]
    demeaned = y - y.groupby(tm).transform("mean")
    lo = demeaned.groupby(tm).transform(lambda s: s.quantile(0.01))
    hi = demeaned.groupby(tm).transform(lambda s: s.quantile(0.99))
    return demeaned.clip(lower=lo, upper=hi)
```
(`src/models.py:make_target`)

The training/validation label is **not** raw `stock_exret`. It is:

$$y_{i,t+1} = \operatorname{clip}\Big(r_{i,t+1} - \bar r_{t+1},\; q_{01}(t+1),\; q_{99}(t+1)\Big)$$

where $\bar r_{t+1}$ is the cross-sectional mean of `stock_exret` across all rows sharing the
same `target_month`, and $q_{01}/q_{99}$ are the 1st/99th percentiles of the **demeaned** series,
again computed within `target_month`.

Two design choices, both economically motivated:

- **Demean within `target_month` (cross-sectional, market-neutral objective).** Subtracting the
  same-month cross-sectional mean removes the common/market component of the realization month
  before the model ever sees it, so $g^*$ is trained to predict *relative* outperformance, not
  the level of market returns. This matches what the strategy actually monetizes: a dollar- and
  beta-neutral long/short book (`src/portfolio.py::optimize_month`) is paid for cross-sectional
  spread, not market direction, so training on a target that has already had the market-month
  effect stripped out aligns the loss function with the trading objective.
- **Clip at the (demeaned) 1st/99th percentile within month (robustness).** Monthly stock returns
  have a heavy right/left tail (halts, M&A jumps, blowups); an unclipped squared-error objective
  would let a handful of extreme rows dominate every month's gradient. Clipping symmetrically at
  1%/99% bounds each month's influence without discarding the row. `tests/test_models.py::
  test_make_target_zero_mean_before_clip` confirms demeaning alone is exactly zero-mean before
  clipping, and that clipping visibly changes the target (i.e. the clip actually engages on
  planted outliers).

Because `.transform("mean")`/`.transform(lambda s: s.quantile(...))` are pandas groupby
transforms, they silently skip `NaN` inputs — so rows with a null `stock_exret` (future exits
with no realized return) neither shift the mean/quantiles used by their same-`target_month`
labelled peers, nor themselves get a spurious label (they stay `NaN` all the way through).
`tests/test_models.py::test_make_target_ignores_null_label_rows` checks this directly by
injecting extra null-label rows and confirming the labelled rows' targets are bit-identical
with or without them.

---

## 2. Training schedule

### 2.1 The rule (`splits`, SPEC §2)

```python
def splits(panel, test_year):
    train_start = config.FIRST_TARGET          # 2015-02-28
    train_end   = Timestamp(year=y-3, month=12, day=31)
    valid_start = Timestamp(year=y-2, month=1,  day=31)
    valid_end   = Timestamp(year=y-1, month=12, day=31)
    test_start  = Timestamp(year=y,   month=1,  day=31)
    test_end    = min(Timestamp(year=y, month=12, day=31), config.TEST_END)  # TEST_END = 2026-08-31
    has_label = panel["stock_exret"].notna()
    train_mask = (tm >= train_start) & (tm <= train_end) & has_label
    valid_mask = (tm >= valid_start) & (tm <= valid_end)          # NOT gated on has_label
    test_mask  = (tm >= test_start)  & (tm <= test_end)           # NOT gated on has_label
```
(`src/models.py:splits`)

**All three masks are built on `target_month`, never `eom`.** This is a direct implementation of
the brief's explicit instruction (p.14): *"Keep labels whose realization falls in the test
period out of training and validation, even if their characteristic month precedes the test
boundary."* Because `target_month = eom + 1`, an `eom`-based cutoff would let the very last
training-eligible row (say `eom = 2020-12-31`) carry a label realized in `target_month =
2021-01-31` — i.e. inside test year 2021's own test window — straight into training. Splitting
on `target_month` instead makes that impossible by construction: the boundary is drawn on the
same calendar axis the label itself lives on, so no realized return from the test window can
ever appear as a training or validation label, regardless of which `eom` produced it.

**Why the training window requires a label but validation/test do not:** fitting model
coefficients needs a target, so `train_mask` additionally requires `stock_exret.notna()`.
Valid/test masks impose no such requirement — every panel row in those windows gets a saved
prediction, including rows whose `stock_exret` is null because the stock made its **last**
appearance in the raw data before a return could be realized (a true exit, not a data gap).
`tests/test_models.py::test_train_excludes_null_label_valid_and_test_keep_it` and
`tests/test_integrity.py`'s schedule test both check this. Valid-window null-label rows are kept
per Amendment A11 (SPEC §10): the gate (`fit_gate`) is fit only on the *labelled* subset of
valid, but predictions are still produced and saved for the unlabelled subset too, because the
2019-2020 validation window doubles as the portfolio-penalty calibration set
(`portfolio.calibrate`), which needs a prediction for every candidate name regardless of whether
it later has a realized return.

### 2.2 Full schedule table, 2021-2026

Train always starts at `config.FIRST_TARGET = 2015-02-28` (the first `target_month` the raw data
can supply, since characteristics begin January 2015 and `ret_exc_lead1m` needs one more month —
brief p.14: *"do not invent December 2014 predictors"*). The inner holdout is the **last 12
target months of train** (`INNER_HOLDOUT_MONTHS = 12`, used by `_fit_specialist`, not part of
`splits` itself — see §5). Test truncates at `config.TEST_END = 2026-08-31` for the final year.

| test_year | train target months | train (n months) | inner holdout (train tail) | valid target months | valid (n months) | test target months | test (n months) |
|---|---|---|---|---|---|---|---|
| 2021 | 2015-02 .. 2018-12 | 47 | 2018-01 .. 2018-12 | 2019-01 .. 2020-12 | 24 | 2021-01 .. 2021-12 | 12 |
| 2022 | 2015-02 .. 2019-12 | 59 | 2019-01 .. 2019-12 | 2020-01 .. 2021-12 | 24 | 2022-01 .. 2022-12 | 12 |
| 2023 | 2015-02 .. 2020-12 | 71 | 2020-01 .. 2020-12 | 2021-01 .. 2022-12 | 24 | 2023-01 .. 2023-12 | 12 |
| 2024 | 2015-02 .. 2021-12 | 83 | 2021-01 .. 2021-12 | 2022-01 .. 2023-12 | 24 | 2024-01 .. 2024-12 | 12 |
| 2025 | 2015-02 .. 2022-12 | 95 | 2022-01 .. 2022-12 | 2023-01 .. 2024-12 | 24 | 2025-01 .. 2025-12 | 12 |
| 2026 | 2015-02 .. 2023-12 | 107 | 2023-01 .. 2023-12 | 2024-01 .. 2025-12 | 24 | 2026-01 .. 2026-08 | 8 |

(Month counts derived directly from the date arithmetic above; the 2021-row figures reproduce the
brief's own worked example on p.14 — train 2015-2018, valid 2019-2020, test 2021 — and the 2022
row reproduces its "next forecast year" example — train through 2019, valid 2020-2021, test
2022.) The union of test target months across all six years is 2021-01 .. 2026-08, 68 months,
matching SPEC §2 and checked byte-for-byte by `tests/test_models.py::
test_splits_schedule_matches_spec` and `tests/test_integrity.py::
test_schedule_and_inner_holdout` (both on synthetic calendars covering 2015-01..2026-07).

**What each window is used for:**
- **Train** fits every model's coefficients (linear baseline weights, LightGBM trees). Only the
  labelled rows participate, since a label is required.
- **Inner holdout** (last 12 months of train, computed inside `_fit_specialist`, not by
  `splits`) selects LightGBM's `num_leaves`/number of boosting rounds — never the official valid
  window (Amendment A2, detailed in §5).
- **Validation** (24 months) is used for exactly two things: (a) the linear baselines' penalty
  ($\alpha$) selection (`fit_baselines`, "template baselines", A2 explicitly permits this), and
  (b) fitting the ridge regime gate (`fit_gate`). It is never used to re-estimate any model's
  coefficients.
- **Test** is held out from every fitting/tuning step; predictions are made on it and scored
  only in `r2_table`/downstream evaluation, never fed back.

**Why no refit on validation:** the brief's own training-procedure wording (p.13) is "*Tune
hyperparameters using a validation sample, estimate model coefficients from training data, and
reserve testing data for out-of-sample evaluation*" — coefficients come from train, validation is
for hyperparameters only, test is untouched until scoring. The repo goes one step further than
this literal instruction (Amendment A2, SPEC §10): even hyperparameter tuning for the five
characteristic specialists, the text specialist and `lgbm_all` is moved off the official 24-month
valid window and onto an *inner* holdout carved out of train itself. This means the 24-month
valid window is genuinely never seen by **any** of the six specialists that make up the headline
`pred_ew` blend, nor by `lgbm_all` — only the gate and the linear baselines' $\alpha$ selection
touch it — which is what lets Amendment A12's validation-window pseudo-OOS diagnostic (§6) be a
meaningful, un-contaminated test of the gate specifically.

**Annual refit, monthly forecasts:** each `test_year` in the loop (`run_all`,
`src/models.py:run_all`) triggers exactly one fit of every model (baselines, `lgbm_all`, the five
specialists, the text specialist, the gate) on that year's train/inner-holdout/valid windows.
The resulting fixed models are then applied, unchanged, to predict every formation month inside
that year's test window (12 months, or 8 for 2026) — i.e. "refitting annually and forecasting
monthly" (brief p.14) is implemented literally as one `run_all` loop iteration per `test_year`,
with `_predict_both` (`src/models.py:_predict_both`) producing the whole valid/test prediction
array from a single fitted object.

**Rows with null labels, one more time, precisely:** excluded from `train_mask` (can't fit
without a label); included in `valid_mask` and `test_mask` regardless of label (A11) — predictions
are made and saved for them, but they are dropped by `r2_table` (`df[... & preds["stock_exret"]
.notna()]`) since there is no realized return to score a forecast against. Only the valid rows
from `test_year == config.TEST_YEARS[0]` (2021, i.e. the 2019-2020 window) are ever written to
`preds.parquet`'s `split == "valid"` rows (`run_all`, `if test_year == config.TEST_YEARS[0]:
frames.append(_mk(valid, "valid"))`) — every other test_year's valid predictions are computed
and consumed internally (by `fit_gate` and the linear-baseline $\alpha$ search) but not persisted,
since only the first year's valid window is needed downstream for portfolio-penalty calibration.

---

## 3. Feature sets

### 3.1 The 147 characteristics + 16 miss_ flags

`config.load_char_list()` reads `factor_char_list.csv` and asserts exactly 147 names. Every
characteristic is rank-transformed within `eom` to $[-1,1]$ (`2\cdot(\text{rank}-1)/(n-1)-1$,
average ties, `NaN \to 0`; `src/data.py:_rank_within_eom`) before it reaches `models.py`. On top
of the 147 ranked characteristics, `src/data.py:_build` adds `miss_<char>` int8 flags for any
characteristic whose null rate — measured **only** on universe rows with `eom <=
config.MISS_FLAG_CUTOFF` (2018-12-31) and required to exceed `config.MISS_FLAG_RATE` (0.20) — is
high enough to be informative on its own (a stock reporting no R&D is economically different from
one with R&D = 0). On the actual cached panel (`outputs/cache/panel.parquet`) this selects
exactly 16 flags: `miss_aliq_mat, miss_debt_gr3, miss_dsale_dinv, miss_dsale_dsga, miss_f_score,
miss_intrinsic_value, miss_inv_gr1, miss_ope_be, miss_ope_bel1, miss_pi_nix, miss_rd5_at,
miss_rd_me, miss_rd_sale, miss_ret_60_12, miss_seas_2_5an, miss_seas_2_5na` — confirming the "16
miss_ flags" figure. `feature_columns(panel)` (`src/data.py:feature_columns`) returns the 147
chars followed by the (16) `miss_` flags present, in stable order — 163 model features in total.
Each `miss_<c>` flag is assigned to the same `CHAR_GROUPS` bucket as its underlying characteristic
`c` (`run_all`'s `char_to_group.get(c[5:] if c.startswith("miss_") else c)`), so a specialist sees
its group's characteristics *and* their missingness flags together.

### 3.2 CHAR_GROUPS — five characteristic specialists (Amendment A9)

`CHAR_GROUPS` (`src/models.py:CHAR_GROUPS`) partitions all 147 characteristics exactly once
(`tests/test_models.py::test_char_groups_partition_all_147` checks no omissions/duplicates)
into five economically-motivated groups. Amendment A9 (SPEC §10) records the rationale: the split
follows the standard factor taxonomy contrasted in Fama-French 5 (RMW profitability vs. CMA
investment), Hou-Xue-Zhang's q-factor model (ROE vs. investment-to-assets I/A), and
Stambaugh-Yuan's mispricing factors (PERF vs. MGMT) — i.e. *profitability/quality* and
*investment/growth* are economically distinct return drivers and are kept as separate
specialists rather than merged, and `ebit_bev`/`sale_bev` were moved from `value` into `quality`
because they are profitability/turnover ratios on book enterprise value, not price multiples.

**value** (16: fundamentals-to-price ratios — cheapness):
`be_me, at_me, ebitda_mev, fcf_me, div12m_me, debt_me, bev_mev, eqpo_me, intrinsic_value, ni_me,
sale_me, eq_dur, eqnpo_me, netdebt_me, ocf_me, rd_me`

**momentum** (14: past-return continuation, reversal and seasonality):
`prc_highprc_252d, resff3_12_1, resff3_6_1, ret_1_0, ret_12_1, ret_12_7, ret_3_1, ret_6_1,
ret_60_12, ret_9_1, seas_1_1an, seas_1_1na, seas_2_5an, seas_2_5na`

**quality** (50: profitability, earnings quality/persistence, safety — FF5's RMW / HXZ's ROE /
Stambaugh-Yuan's PERF side; plus `ebit_bev`, `sale_bev` moved in by A9):
`aliq_at, aliq_mat, at_be, at_turnover, cash_at, cop_at, cop_atl1, dgp_dsale, dsale_dinv,
dsale_drec, dsale_dsga, earnings_variability, ebit_sale, f_score, gp_at, gp_atl1, kz_index,
mispricing_perf, ni_ar1, ni_be, ni_inc8q, ni_ivol, niq_at, niq_at_chg1, niq_be, niq_be_chg1,
niq_su, o_score, ocf_at, ocf_at_chg1, ocfq_saleq_std, op_at, op_atl1, ope_be, ope_bel1, opex_at,
pi_nix, qmj, qmj_growth, qmj_prof, qmj_safety, rd_sale, rd5_at, sale_emp_gr1, saleq_su,
tangibility, tax_gr1a, z_score, ebit_bev, sale_bev`

**investment_growth** (37: asset/investment growth, accruals, net issuance — FF5's CMA / HXZ's
I/A / Stambaugh-Yuan's MGMT side):
`at_gr1, be_gr1a, capex_abn, capx_gr1, capx_gr2, capx_gr3, coa_gr1a, col_gr1a, cowc_gr1a,
fnl_gr1a, lnoa_gr1a, lti_gr1a, ncoa_gr1a, ncol_gr1a, nfna_gr1a, nncoa_gr1a, noa_gr1a, sti_gr1a,
ppeinv_gr1a, debt_gr3, emp_gr1, inv_gr1, inv_gr1a, sale_gr1, sale_gr3, saleq_gr1, chcsho_12m,
dbnetis_at, eqnetis_at, eqnpo_12m, netis_at, oaccruals_at, oaccruals_ni, taccruals_at,
taccruals_ni, noa_at, mispricing_mgmt`

**risk_liquidity** (30: size, liquidity, volatility, beta, skewness, trading activity):
`age, ami_126d, beta_60m, beta_dimson_21d, betabab_1260d, betadown_252d, bidaskhl_21d,
corr_1260d, coskew_21d, dolvol_126d, dolvol_var_126d, iskew_capm_21d, iskew_ff3_21d,
iskew_hxz4_21d, ivol_capm_21d, ivol_capm_252d, ivol_ff3_21d, ivol_hxz4_21d, market_equity, prc,
rmax1_21d, rmax5_21d, rmax5_rvol_21d, rskew_21d, rvol_21d, turnover_126d, turnover_var_126d,
zero_trades_126d, zero_trades_21d, zero_trades_252d`

$16+14+50+37+30 = 147$. With their `miss_` flags folded in by group, the five specialists train
on $18, 17, 59, 39, 30$ features respectively (summing to $163$), computed by mapping each of the
16 `miss_` flags to its underlying characteristic's `CHAR_GROUPS` bucket: `miss_intrinsic_value`
and `miss_rd_me` add 2 to value (16→18); `miss_ret_60_12`, `miss_seas_2_5an`, `miss_seas_2_5na` add
3 to momentum (14→17); `miss_aliq_mat`, `miss_dsale_dinv`, `miss_dsale_dsga`, `miss_f_score`,
`miss_ope_be`, `miss_ope_bel1`, `miss_pi_nix`, `miss_rd_sale`, `miss_rd5_at` add 9 to quality
(50→59); `miss_debt_gr3` and `miss_inv_gr1` add 2 to investment_growth (37→39); no `miss_` flag's
underlying characteristic falls in risk_liquidity, so it stays at its base 30.

### 3.3 Text feature set (the sixth specialist)

`text.TEXT_FEATURES` (`src/text.py:TEXT_FEATURES`) is the model-feature list used by the text
specialist — 16 columns when the FinBERT score cache is present:

- `n_filings` — count of 8-K filings for that (permno, eom).
- `item_1_01, item_1_02, item_2_01, item_2_02, item_2_05, item_2_06, item_3_01, item_4_01,
  item_4_02, item_5_02, item_7_01, item_8_01` — 12 counts, one per `config.KEY_ITEMS` (Form 8-K
  item codes: material agreements, asset acquisition/disposition, results of operations,
  exit/impairment costs, delisting notice, accountant/restatement changes, officer/director
  changes, Reg FD disclosure, other events).
- `tone_mean, tone_min, fb_neg_max` — FinBERT sentiment aggregates (`fb_pos - fb_neg` mean/min,
  and max `fb_neg`), present only when `text.scores_path_for(text.MAX_LENGTH)` exists.

`has_filing` is deliberately **excluded** from `TEXT_FEATURES` (Amendment A1): this 8-K archive
is retrospectively assembled, so same-month filing coverage strongly encodes forward survival
(re-measured: 1.3% same-month coverage for a permno's *last* panel row vs. 59.7% for survivors to
the panel's end — SPEC §10 A1, corrected 2026-09-27). `has_filing` is carried only as an
auxiliary column so `models.py` can restrict the text specialist to filer rows and neutralize
non-filers' forecast to exactly zero (§6).

---

## 4. Baselines: OLS, Ridge, Lasso, ElasticNet

`fit_baselines` (`src/models.py:fit_baselines`) fits four linear models on the full 163-feature
set, using `sklearn`'s default `fit_intercept=True` (unlike the legacy
`penalized_linear_hackathon.py` template, which manually demeans $y$ and fits with
`fit_intercept=False` — here the intercept is left to the solver, which is equivalent in effect
since `make_target` has already demeaned $y$ within `target_month` before it ever reaches
`fit_baselines`). Using `sklearn`'s own parameterizations:

$$\text{OLS:}\quad \min_{\beta_0,\beta} \sum_{i} \left(y_i - \beta_0 - x_i^\top\beta\right)^2$$

$$\text{Ridge}(\alpha):\quad \min_{\beta_0,\beta} \left\|y - \beta_0\mathbf 1 - X\beta\right\|_2^2 + \alpha\|\beta\|_2^2$$

$$\text{Lasso}(\alpha):\quad \min_{\beta_0,\beta} \frac{1}{2n}\left\|y - \beta_0\mathbf 1 - X\beta\right\|_2^2 + \alpha\|\beta\|_1$$

$$\text{ElasticNet}(\alpha,\rho{=}0.5):\quad \min_{\beta_0,\beta} \frac{1}{2n}\left\|y - \beta_0\mathbf 1 - X\beta\right\|_2^2 + \alpha\rho\|\beta\|_1 + \frac{\alpha(1-\rho)}{2}\|\beta\|_2^2$$

(the $\rho = 0.5$ `l1_ratio` for ElasticNet is fixed, not searched; `Lasso`/`ElasticNet` use
`max_iter=5000`.) OLS has no penalty and is fit once, directly, on the full training window —
it never touches validation at all (confirmed by `tests/test_models.py::
test_run_all_ignores_valid_labels`'s explicit "OLS has no alpha" comment).

**Alpha grids** (`src/models.py`, exact values via `np.logspace`):
- `RIDGE_ALPHAS = np.logspace(-3, 6, 19)`: 0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10,
  31.6, 100, 316, 1000, 3160, 10⁴, 3.16×10⁴, 10⁵, 3.16×10⁵, 10⁶.
- `LASSO_ALPHAS = ENET_ALPHAS = np.logspace(-8, -1, 15)`: 10⁻⁸, 3.16×10⁻⁸, 10⁻⁷, 3.16×10⁻⁷, 10⁻⁶,
  3.16×10⁻⁶, 10⁻⁵, 3.16×10⁻⁵, 10⁻⁴, 3.16×10⁻⁴, 10⁻³, 3.16×10⁻³, 10⁻², 3.16×10⁻², 10⁻¹.

**Selection** (`_select_alpha`, `src/models.py:_select_alpha`): for each candidate $\alpha$, fit
on the (possibly subsampled) training window and score `mean_squared_error` on the **labelled**
validation rows; keep the $\alpha$ with lowest validation MSE. `_warn_if_boundary` prints a
warning if the chosen $\alpha$ sits at either end of its grid (a sign the grid should be widened).
This valid-MSE alpha search is explicitly permitted by Amendment A2 because Ridge/Lasso/
ElasticNet/OLS are "template baselines" — not inputs to the gate or to `pred_ew` — so touching
the official validation window for their hyperparameter is acceptable, unlike the specialists
(§5, §2.2).

**Subsampling then full refit:** `MAX_ROWS_PENALIZED = 100_000` caps the *search* for Lasso and
ElasticNet only (`_PENALIZED_SPECS`'s per-model `max_rows` entry: `None` for Ridge, `100_000` for
Lasso/ElasticNet) — coordinate-descent fitting at `max_iter=5000` across up to 15 alphas is
expensive on a training window that can exceed 100k rows for later test years, so the search
draws a fixed `np.random.RandomState(config.SEED)` subsample of at most 100,000 rows. Ridge's
search is never subsampled since its closed-form solve is cheap regardless of $n$. Critically,
**the final refit at the chosen $\alpha$ always uses the full (non-subsampled) training window**
for all three penalized models (`out[name] = cls(alpha=alpha, **kwargs).fit(Xtr, ytr)` uses the
untouched `Xtr`) — subsampling changes only which rows influence which $\alpha$ is picked, never
what data the returned model is actually fit on.

**`lgbm_all`:** a single LightGBM model trained on the entire 163-feature set (not restricted to
one `CHAR_GROUPS` bucket), tuned via the same IC-on-inner-holdout procedure as the five
specialists (`_fit_specialist(train, ytr, feature_cols, label=f"lgbm_all y={test_year}")` —
§5). It is reported in `r2_table`/`PRED_COLS` and used as an ablation backtest signal
(`MAIN.py` step 6), but is not one of the six specialists blended into `pred_ew`.

---

## 5. LightGBM specialists

### 5.1 What gradient boosting does (brief intuition)

LightGBM fits an additive ensemble of shallow regression trees: $F_m(x) = F_{m-1}(x) +
\nu\, h_m(x)$, where each new tree $h_m$ is fit to approximate the negative gradient of the loss
with respect to the current ensemble's prediction — for squared-error regression (LightGBM's
`objective="regression"`) that gradient reduces to the current residual $y - F_{m-1}(x)$, and
$\nu$ = `learning_rate` shrinks each tree's contribution so no single tree dominates the fit.
LightGBM specifically grows trees **leaf-wise** (splits the single leaf with the largest loss
reduction next, rather than expanding every leaf at a given depth), so `num_leaves` — not tree
depth — is the primary complexity control, and it bins continuous features into histograms for
fast split search.

### 5.2 Every hyperparameter (`_lgbm_base_params`, `src/models.py:_lgbm_base_params`)

| parameter | value | why |
|---|---|---|
| `objective` | `"regression"` (squared error) | matches the continuous, demeaned/clipped target from `make_target` |
| `learning_rate` | `0.05` | shrinkage; small enough that hundreds of rounds are needed, which is exactly what `ROUND_GRID` searches |
| `min_child_samples` | `1000` | minimum rows per leaf — strong regularization forcing coarse, well-populated splits given a monthly cross-section that can run to thousands of stocks |
| `feature_fraction` | `0.5` | randomly samples half the 163 (or per-group) features per tree — decorrelates trees, reduces overfitting |
| `bagging_fraction` / `bagging_freq` | `0.8` / `1` | resamples 80% of rows every iteration — stochastic gradient boosting, further variance reduction |
| `lambda_l2` | `10` | L2 penalty on leaf output values — shrinks predicted leaf values toward zero, an explicit ridge-style penalty on the tree outputs |
| `seed` | `config.SEED` (42) | read at *call* time, not import time, so tests can monkeypatch it |
| `n_jobs` | `config.N_JOBS` (6) | ditto — some tests force `N_JOBS=1` for bit-exact determinism, since LightGBM's multi-threaded histogram construction is not guaranteed to sum floats in the same order across thread counts |
| `verbose` | `-1` | silence LightGBM's own logging |
| `num_leaves` | grid `{7, 15}` | small-tree grid, the actual complexity knob being searched |
| `num_boost_round` | grid `ROUND_GRID = (100, 200, 300, 500, 700, 1000)` | Amendment A13's fixed rounds grid, floor 100, up to 1000 |

### 5.3 IC-based selection on the inner holdout (Amendment A13) — and the 1-tree bug

```python
def _fit_lgbm_tuned(Xtr, ytr, Xva, yva, months_va):
    for num_leaves in (7, 15):
        booster = lgb.train({**base_params, "num_leaves": num_leaves}, dtr,
                             num_boost_round=max(ROUND_GRID))
        for k in ROUND_GRID:
            pred = booster.predict(Xva, num_iteration=k)
            ic = _ic_mean(pred, yva, months_va)
            if ic > best_ic:
                best_ic, best_leaves, best_rounds = ic, num_leaves, k
```
(`src/models.py:_fit_lgbm_tuned`)

For each of the two `num_leaves` values, LightGBM trains **one** booster for the full 1000
rounds, then `booster.predict(Xva, num_iteration=k)` re-scores it at each `ROUND_GRID`
checkpoint — cheap because boosting is additive, so truncating to the first $k$ trees after
training 1000 is mathematically identical to having trained only $k$ rounds. Each checkpoint's
predictions are scored by **mean monthly Spearman IC** (`_ic_mean`, reusing `monthly_ic`'s
groupby/rank-correlation logic) against `yva` — the *inner holdout* (`Xva`/`yva`/`months_va`
here are the last `INNER_HOLDOUT_MONTHS = 12` target months of the training window, carved out
inside `_fit_specialist`), **never the official 24-month valid window**. The
`(num_leaves, rounds)` pair with the highest inner-holdout mean IC wins.

This is a deliberate departure from SPEC §5's originally-written wording (*"...up to 1000
rounds with early stopping (50) on valid; pick by valid MSE"*), documented as Amendment A13:
pooled-MSE early stopping was found, in validation diagnostics, to select **degenerate**
low-round/low-leaf models — e.g. a single tree with `num_leaves=7` producing only 7 distinct
predicted values across an entire test month. The mechanism is that squared-error, pooled across
a huge, noisy panel of monthly stock returns, can be minimized by a near-constant prediction
(few leaves, few rounds) even though such a model has essentially zero cross-sectional
discriminating power — useless for a long/short strategy that only cares about *within-month
rank*, not the absolute level of the forecast. Switching the selection criterion to mean monthly
Spearman IC — a rank-based, cross-sectional metric computed separately in each holdout month —
directly penalizes this degeneracy, since a near-constant column cannot produce a meaningful
rank correlation. `tests/test_models.py::test_specialist_not_degenerate` is the regression test
for this fix: it asserts every test month has more than 50 distinct `pred_spec_quality` values
on a 150-stock/month synthetic panel with a planted signal.

### 5.4 Refit on full train, and the filer-only text specialist

```python
def _fit_specialist(train_df, y_train, feature_cols, label=None):
    tm = train_df["target_month"].values
    months = np.sort(np.unique(tm))
    ho = min(INNER_HOLDOUT_MONTHS, max(1, len(months) - 1))
    is_ho = np.isin(tm, months[-ho:])
    ...
    num_leaves, num_rounds, ic = _fit_lgbm_tuned(X[~is_ho], y_train[~is_ho], X[is_ho], y_train[is_ho], tm[is_ho])
    dfull = lgb.Dataset(X, label=y_train)
    return lgb.train({**_lgbm_base_params(), "num_leaves": num_leaves}, dfull, num_boost_round=num_rounds)
```
(`src/models.py:_fit_specialist`)

Once `(num_leaves, rounds)` is chosen on the inner holdout, a **new** booster is trained with
those settings on the **entire** training window (inner-train rows plus the inner-holdout rows
folded back in) — the inner holdout is purely a tuning device, not a subset withheld from the
final fit. `run_all` calls `_fit_specialist` once per `CHAR_GROUPS` key (with that group's
feature columns) and once more for `lgbm_all` (all 163 features).

The **text specialist** uses the same `_fit_specialist` machinery but restricted to filer rows
only, per Amendment A1: `_fit_specialist(train.loc[train_filer.values], ytr[train_filer.values],
text_cols, label=...)`. Its predictions are produced by `_predict_filer_only`
(`src/models.py:_predict_filer_only`), which returns `NaN` for every non-filer row (the model
never sees or predicts them) — that `NaN` becomes an exact zero once z-scored (§6).

### 5.5 Seeds, `n_jobs`, determinism

`config.SEED = 42` and `config.N_JOBS = 6` are read at *call* time by `_lgbm_base_params`
(comment: "so tests can monkeypatch them, e.g. to force single-threaded determinism"), not
captured once at import. Production runs use `N_JOBS=6`; several tests
(`test_specialist_ignores_valid_labels`, `test_run_all_ignores_valid_labels`) explicitly force
`config.N_JOBS = 1` before comparing predictions bit-for-bit, because LightGBM's multi-threaded
histogram construction is not guaranteed to sum floating-point values in the same order across
different thread counts — single-threading is the only way those tests can assert exact
(`np.allclose`/`np.array_equal`) reproducibility rather than "close enough."

---

## 6. Combination: from six specialists to one signal

### 6.1 Z-scoring within `eom`

```python
def _zscore_by_eom(values, eom):
    s = pd.Series(values, index=eom.index)
    g = s.groupby(eom)
    z = (s - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
    return z.fillna(0.0)
```
(`src/models.py:_zscore_by_eom`)

Every specialist's raw (return-unit) prediction is standardized **within `eom`** (the row's
formation month — the same key `portfolio.py` groups by when constructing the tradable
cross-section for that month) to a comparable, dimensionless $Z_k(i,t)$: $Z_k = (\hat
y_k - \bar{\hat y}_{k,t})/\sigma(\hat y_{k,t})$. A group whose within-month standard deviation is
exactly zero would divide by `NaN`, which `.fillna(0.0)` turns into an exact zero — the same
mechanism handles the text specialist's non-filer `NaN` predictions: ignored by the group
mean/std (pandas `groupby` skips `NaN`), then set to exactly `0.0`. This single function is
therefore already the "filer-only z-score with a neutral 0 for non-filers" the spec calls for;
`tests/test_models.py::test_zscore_by_eom_zero_for_nan_rows` checks both properties directly.

### 6.2 pred_ew — the headline (Amendment A12)

$$\texttt{pred\_ew}_{i,t} = \frac{1}{6}\sum_{k \in \{\text{value, momentum, quality, investment\_growth, risk\_liquidity, text}\}} Z_k(i,t)$$

(`Z_valid[specialists_all].mean(axis=1)`, `specialists_all = list(CHAR_GROUPS) + ["text"]`).
`pred_ew_notext` is the same average over the five characteristic specialists only (`nt =
list(CHAR_GROUPS)`), an ablation isolating text's marginal contribution.

**Why simple 1/N rather than a fitted combination weight:** SPEC §10 Amendment A12 records the
reasoning explicitly. A ridge regime gate (§6.3) was the *pre-registered* headline
(`docs/research_log.md`, 2026-09-27 pre-registration entry), but a validation-window-only
diagnostic — a split-half pseudo-out-of-sample test *inside* each 24-month validation window
(fit on one 12-month half, evaluate on the other) — found the gate underperforming a plain
equal-weight blend in every test year:

| combiner | mean pseudo-OOS IC (validation-window split-half) |
|---|---|
| gate (full) | −0.012 |
| gate, n-scaled alpha | 0.004 |
| gate without state interactions | 0.009 |
| NNLS combination | 0.025 |
| IC-weighted combination | 0.038 |
| **equal-weight (1/N)** | **0.044** (best in 5 of 6 years, never negative) |

The stated mechanism is an identification problem: the gate design matrix has 18 free parameters
($6$ specialists $\times$ (1 base $+$ 2 `STATE_VARS` interactions) — confirmed directly by the
gate design in §6.3), but only $\sim$24 monthly observations are available to fit them each year;
pooled-MSE fits can flip a specialist's sign after a single bad factor year, and the state
interactions add noise rather than signal. This is the same estimation-risk argument
DeMiguel, Garlappi & Uppal (2009) make for naive 1/N diversification against optimized portfolio
weights under parameter uncertainty — cited directly in the amendment. `pred_ew` was switched to
headline **post-hoc**, after this evidence was seen, and is disclosed as such; the decision itself
rests only on the validation-window table above, never on test-period returns.

For full candor the amendment also discloses that an *interface smoke run* (old, pre-A9,
4-specialist model code that crashed later in portfolio calibration) had already printed
test-period ICs before the A12 decision was made: gate 0.022 vs. equal-weight 0.065
(`docs/smoke_run_oos_r2_2026-09-27.csv`). This table is quoted here only as disclosure of what
was seen and when — the research log is explicit that it was **not** used to justify the switch,
and it predates Amendment A9 (it has only four characteristic specialists, not five) and the
IC-based specialist tuning of A13, so it is not comparable to the current, fully-specified model.

### 6.3 pred_ew_ret — the return-unit counterpart

```python
raw_specs = [preds[f"pred_spec_{k}"] for k in specialists_all]
preds["pred_ew_ret"] = {split: np.mean([np.nan_to_num(r[split], nan=0.0) for r in raw_specs], axis=0)
                         for split in ("valid", "test")}
```
`pred_ew` lives in z-score units (mean/std $\approx 0/1$ within every month by construction), so
comparing it to `stock_exret` (a return) inside `oos_r2 = 1 - \sum(y-\hat y)^2/\sum y^2$ is not
meaningful — the numerator is dominated by the unit mismatch, not by predictive skill.
`pred_ew_ret` exists solely to give the headline blend a return-denominated proxy: the row-mean
of the six specialists' **raw** (unstandardized) forecasts, with the text specialist's non-filer
`NaN` treated as `0` via `np.nan_to_num` (same convention as the z-score fillna). This is the
number `MAIN.py` prints as "headline OOS R2 (pred_ew_ret, the return-unit version of pred_ew,
A12)", and it is the only column that lets `r2_table` report a numerically sensible $R^2$ for the
headline strategy. `tests/test_models.py::test_pred_ew_ret_is_raw_specialist_mean` pins the exact
formula.

### 6.4 The regime gate (kept as an ablation/explainability exhibit)

```python
def _build_gate_design(Z, S):
    cols = {k: Z[k].values for k in Z.columns}
    for k in Z.columns:
        for j in S.columns:
            cols[f"{k}__{j}"] = Z[k].values * S[j].values
    return pd.DataFrame(cols, index=Z.index)
```

Design matrix: $[Z_k]_{k=1}^{6} \cup [Z_k \cdot s_j]_{k=1..6,\, j=1,2}$ — the six specialist
Z-scores plus their interaction with two standardized market-state variables, `config.STATE_VARS
= ['mkt_vol12', 'disp']` (12-month trailing market volatility and cross-sectional return
dispersion, both from `data.market_state()`). $6 + 6\times2 = 18$ columns, matching the "18
parameters" cited in A12.

**State standardization:** for a given `test_year`, $s_j$ is standardized using the mean/std of
`state` computed **only over that year's own 24 valid-window `eom` values**
(`state.loc[state.index.isin(valid_eoms)]`, `mu, sd = state_valid.mean(), state_valid.std()`),
then the *same* `mu`/`sd` are applied out-of-sample to standardize the test window's state — an
ordinary train-only scaler pattern, not leakage, since the valid window always precedes the test
window in time.

**Fitting:** `RidgeCV(alphas=_gate_alphas(n), fit_intercept=False, cv=GroupKFold(n_splits=min(6,
n_groups)).split(..., groups=eom_valid_lab))` on the labelled valid rows only, target = the same
`make_target` label. `_gate_alphas(n) = n * np.logspace(-3, 2, 11)` — 0.001n, 0.00316n, ..., 100n
— scaled by the labelled-row count $n$ because sklearn's `Ridge` objective is
$\|y-Xw\|_2^2 + \alpha\|w\|_2^2$ **without** a $1/n$ normalizer, so an unscaled grid would be
comparable across years only by coincidence (valid-window row counts differ every year, per
§2.2's month counts times the universe size); scaling by $n$ keeps the effective penalty per row
roughly constant. `cv` groups folds by whole `eom` months (not individual rows) via `GroupKFold`,
so no fold ever splits a single month's cross-section across train/test — with 24 valid months
there are always $\geq 2$ groups, and the code caps folds at $\min(6, n_{\text{groups}})$.
`fit_intercept=False` is appropriate since both $Z_k$ (z-scored) and $y$ (target-month-demeaned)
are already approximately zero-mean.

**Why it failed on validation:** see the pseudo-OOS table in §6.2 — the gate's failure is a
combiner-identification problem (18 parameters, ~24 observations), not a bug in any individual
specialist. `pred_gate_notext` (gate fit over the five characteristic specialists only, dropping
the text interaction terms) is a separate fit, tested the same way and similarly not promoted.

**Kept as an ablation/explainability exhibit:** `MAIN.py` step 6 still backtests `pred_gate` and
`pred_gate_notext` among the fixed ablations (never as headline), and `evaluate.py`'s regime chart
(`gate effective weights over time`, plotting $b_k + \sum_j c_{kj}\, s_j(t)$ per specialist $k$)
uses the saved gate coefficients purely for interpretability — showing *how* a state-conditioned
combiner would have shifted specialist weight with volatility/dispersion, without using that
combiner to size the actual book.

---

## 7. Evaluating the forecasts

### 7.1 OOS R2 — zero benchmark (brief p.14)

```python
def oos_r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)
```
(`src/models.py:oos_r2`) implements exactly

$$R^2_{OOS} = 1 - \frac{\sum_{(i,t)\in\mathcal T}\left(r_{i,t+1}-\hat r_{i,t+1}\right)^2}{\sum_{(i,t)\in\mathcal T} r_{i,t+1}^2}$$

The brief is explicit that the denominator is **not** de-meaned by the historical average return
(the usual OOS $R^2$ convention) — the implicit benchmark is a forecast of exactly zero, i.e. "no
predictability at all," because a historical-mean-based portfolio would actually underperform
random stock selection. Any positive $R^2$, however small, is genuine out-of-sample skill by this
yardstick; the brief notes published OOS $R^2$ for stock returns typically runs 1-2% even for
complex models. `r2_table` additionally reports `oos_r2_demeaned` (the same formula against
`stock_exret` minus its within-`eom` cross-sectional mean) — a diagnostic beyond the brief's own
figure, useful for separating cross-sectional skill from any residual common-month effect, but the
brief's literal $R^2$ is the raw-`stock_exret` column.

### 7.2 Why z-scored signals get IC only

`pred_ew` and `pred_ew_notext` are z-score-valued, not return-valued, so `R2_COLS` explicitly
excludes them (`R2_COLS = [c for c in PRED_COLS if c not in ("pred_ew", "pred_ew_notext")]`) — an
$R^2$ computed against `stock_exret` for a z-scored signal is a pure unit mismatch, not a
meaningful diagnostic. In the disclosed smoke-run table (§6.2's provenance caveats apply)
`pred_ew`'s `oos_r2` is $-13.8$ and `pred_ew_notext`'s is $-20.1$ — large negative numbers that
simply reflect $\sum(y-\hat y)^2 \gg \sum y^2$ once a $\sim N(0,1)$ signal is compared to a
return series with a much smaller variance, not a statement about forecast quality. IC is
scale-invariant (a rank correlation), so it remains meaningful for these two columns even though
$R^2$ does not — `tests/test_models.py::test_r2_table_nan_for_ew_columns` checks precisely this
split (NaN $R^2$, non-NaN IC) on synthetic data.

### 7.3 Monthly Spearman IC and its t-stat

```python
def monthly_ic(df, col):
    def _ic(g):
        g = g.dropna()
        if len(g) < 5:
            return np.nan
        return spearmanr(g[col], g["stock_exret"])[0]
    return df.groupby("eom")[[col, "stock_exret"]].apply(_ic)
```
(`src/models.py:monthly_ic`) computes the Information Coefficient — Spearman rank correlation
between a forecast and the realized `stock_exret` — separately within each `eom`, skipping months
with fewer than 5 valid rows. In `r2_table`:

$$t_{IC} = \frac{\overline{IC}}{s_{IC}/\sqrt{n}}$$

a plain one-sample $t$-statistic against $H_0: E[IC]=0$, using the across-month sample mean
$\overline{IC}$, sample standard deviation $s_{IC}$ (`ddof=1`) and $n$ = number of non-NaN monthly
ICs. This is *not* Newey-West adjusted — the Newey-West (3-lag) correction appears elsewhere, in
`evaluate.py`'s portfolio-return alpha/beta regression (SPEC §7), a different, downstream
statistic over realized portfolio returns, not this per-model forecast diagnostic.

### 7.4 `r2_table` columns

`r2_table(preds)` (`src/models.py:r2_table`) filters to `split == "test"` rows with non-null
`stock_exret`, then for every `PRED_COLS` entry with at least one non-null prediction, emits one
row of: `model, oos_r2, oos_r2_demeaned, mean_ic, ic_tstat` (`oos_r2`/`oos_r2_demeaned` are `NaN`
for `pred_ew`/`pred_ew_notext` per §7.2; `mean_ic`/`ic_tstat` are computed for every model,
including those two).

---

## 8. `run_all` outputs, caching, runtime

### 8.1 `preds.parquet`

`run_all` (`src/models.py:run_all`) writes `config.CACHE_DIR / 'preds.parquet'` with columns
`permno, eom, target_month, stock_exret, test_year, split` plus all 16 `PRED_COLS`:
`pred_ols, pred_ridge, pred_lasso, pred_enet, pred_lgbm_all, pred_spec_value,
pred_spec_momentum, pred_spec_quality, pred_spec_investment_growth, pred_spec_risk_liquidity,
pred_spec_text, pred_gate, pred_gate_notext, pred_ew_ret, pred_ew, pred_ew_notext` — 22 columns
total, checked byte-for-byte by `tests/test_models.py::test_run_all_schema`. `split` is `"test"`
for every `test_year`'s test window, and `"valid"` only for `test_year == config.TEST_YEARS[0]`
(2021, i.e. the 2019-2020 window used downstream for portfolio-penalty calibration — §2.2).

### 8.2 `gate_coefs.parquet`

`gate_coefs_frame` (`src/models.py:gate_coefs_frame`) writes one row per `(specialist, term)`
pair, per `test_year`, for the **with-text** gate only (`pred_gate_notext`'s own coefficients are
never appended to `all_gate` — only its predictions are saved in `preds.parquet`):
`test_year, specialist, term, coef, state_mean_mkt_vol12, state_std_mkt_vol12, state_mean_disp,
state_std_disp`. `term` is `"base"` for the plain $Z_k$ coefficient or the `STATE_VARS` name
(`"mkt_vol12"`/`"disp"`) for that specialist's interaction coefficient — 3 rows per specialist
(1 base + 2 interactions) $\times$ 6 specialists = 18 rows per `test_year`, i.e. 108 rows total
across the six years. The `state_mean_*`/`state_std_*` columns repeat that year's $(\mu,\sigma)$
across all 18 rows — redundant but convenient for reconstructing the "effective weight"
$b_k + \sum_j c_{kj}\,s_j(t)$ chart (SPEC §7) without a second lookup.

### 8.3 Caching and the `--reuse-preds` staleness guard

`MAIN.py`'s `--reuse-preds` flag (lines 77-91) loads the cached `preds.parquet` instead of
re-running `models.run_all()`, behind two guards:

1. **Schema guard:** `set(models.PRED_COLS) - set(preds.columns)` must be empty, or `MAIN.py`
   raises `RuntimeError` naming the missing columns — protects against reusing a cache built by
   an older `models.py` whose `PRED_COLS` list was shorter (e.g. pre-A9's four-specialist code).
2. **Staleness guard:** `preds_path.stat().st_mtime < Path(models.__file__).stat().st_mtime`
   raises `RuntimeError` if the cached parquet is older than `src/models.py` itself — any edit to
   `models.py` after the last `run_all()` invalidates `--reuse-preds` until the cache is rebuilt.
   This is a file-mtime check on `models.py` specifically, not a content hash and not a check on
   its dependencies (`src/data.py`, `src/text.py`, `src/config.py`); editing one of those without
   touching `models.py`'s own mtime would not trip the guard.

### 8.4 Runtime expectations

No full-scale (all six test years, full ~2,720-stocks/month universe) timing has been measured or
disclosed in the sources available for this chapter — none should be inferred or invented here.
The one disclosed runtime figure is a much smaller benchmark: `tests/test_integrity.py`'s own
comment on the production hyperparameter grids ("`ROUND_GRID` up to 1000 rounds $\times$ 2 leaf
configs $\times$ 7 model fits" plus the 15-candidate Lasso/ElasticNet alpha grids) states these
take roughly 20-25 minutes for a **single** `run_all()` call restricted to **one** test year on a
600-permno subset of the real panel — which is why that suite's own mini end-to-end tests
monkeypatch `ROUND_GRID`, `RIDGE_ALPHAS`, `LASSO_ALPHAS`, `ENET_ALPHAS` and
`MAX_ROWS_PENALIZED` down to much smaller grids purely for test speed (explicitly documented as
"a mechanics/interface smoke test... not a performance benchmark"). A full six-year run over the
complete universe (~5,756 permnos, up to 107 training months for 2026) is a substantially larger
computation than this benchmark and has not itself been timed in any source read for this
chapter.

A second, separately disclosed data point: the interface smoke run (the same crashed, pre-A9 run
discussed in §0.3/§6.2 — older code with 4 characteristic specialists, MSE-based early stopping,
and no tone features) timed the full `models.run_all()` call at 2,860 seconds (~48 minutes) on
this laptop, with per-year timings ranging 295–872 seconds, while other jobs were sharing the CPU
concurrently. That figure is not comparable to the current code: the current code trains up to
1,000 boosting rounds $\times$ 2 `num_leaves` settings per specialist (six specialists plus
`lgbm_all`) and then refits at the chosen setting, all against a five-specialist (not
four-specialist) `CHAR_GROUPS` split, so a full current-code run should be expected to take longer
than 48 minutes on this same laptop; it should be far faster on the DGX's 20 cores with
`N_JOBS` raised accordingly.

---

## 9. Tests

### 9.1 `tests/test_models.py`

Fast (no `@pytest.mark.slow`) unit tests cover, on small synthetic panels: `CHAR_GROUPS`
partitioning all 147 characteristics with no gaps/duplicates and A9's five-group/`ebit_bev`
+`sale_bev`-in-quality shape; the exact per-year schedule and inner-holdout structure; that
training excludes null-label rows while valid/test keep them (A11); `make_target`'s zero-mean-
before-clip property and its null-row invariance; the `oos_r2` formula; `_zscore_by_eom`'s
filer-only-zero behavior; that corrupting valid-window labels never changes a single specialist's
test-set predictions (`test_specialist_ignores_valid_labels`, and the `run_all`-level version
`test_run_all_ignores_valid_labels`, which also checks that Ridge/Lasso/ElasticNet/the gate *do*
change — since A2 explicitly allows those to see valid); that valid predictions are produced for
null-label rows too (A11); the full `preds.parquet`/`gate_coefs.parquet` schema and row counts on
a planted-signal synthetic run (`synthetic_run` fixture, `test_run_all_schema`,
`test_planted_signal_specialist_and_gate`, `test_specialist_not_degenerate` — the A13 1-tree-bug
regression check, `test_r2_table_nan_for_ew_columns`, `test_pred_ew_ret_is_raw_specialist_mean`).

Two `@pytest.mark.slow` tests touch real data or need many repetitions:
`test_real_panel_ignores_test_labels` corrupts test-split labels on a 400-permno real-panel
subset and checks every one of the 16 `PRED_COLS` is bit-identical, since test labels are used
only for later evaluation, never fitting/prediction. `test_shuffled_label_ic_small` (discussed
together with its integrity-suite counterpart below) is the module-level shuffled-label null
check.

### 9.2 `tests/test_integrity.py` — schedule and shuffled-label tests

`test_schedule_and_inner_holdout` reproduces the per-year train<valid<test ordering and the
68-month test-union check on a synthetic calendar, and additionally checks the inner-holdout
property `_fit_specialist` relies on but `splits` itself doesn't expose: the last 12 target
months of train sit strictly inside train, at its tail (`inner_train_months.max() <
holdout_months.min()`, `holdout_months.max() == months.max()`), for every one of the six test
years.

**Why the shuffled-label test averages 6 independent reps with a statistical bound, rather than
one fit against a fixed threshold:** shuffling `stock_exret` within `eom` for train+valid rows
destroys the genuine label relationship, so the expected test-year IC of any model fit on the
shuffled data should be zero. But a *single* fit can still land a modest, non-trivial spurious IC
(observed in practice: $-0.0229$ and $+0.0383$ in two independent single-rep runs of the full
`run_all()` pipeline — inconsistent sign, both individually plausible as either noise or a small
real leak) simply because gradient boosting can overfit its inner-holdout noise onto one
specialist/feature by pure chance; that per-fit event does not shrink with a larger universe,
because it is a property of a single fit, not of sampling variance within it.
`test_shuffled_label_ic_real` (`tests/test_integrity.py`) fixes this by treating $R=6$
*independent* reps (independent shuffle **and** independent LightGBM seed per rep) of the full
`run_all()` pipeline's mean test-year `pred_ew` IC as an i.i.d. sample of a mean-zero null, and
testing the **sample mean** against a bound scaled by the *observed* between-rep spread rather
than a fixed number:

$$|\overline{IC}_{\text{6 reps}}| < 2.5\cdot\frac{s_{IC}}{\sqrt{6}} + 0.005$$

a loose, two-sided $\sim$99% $t$-style bound on the mean, with a `+0.005` floor so the bound
cannot collapse toward zero purely because six reps happened to have a small observed spread.
The module-level `test_shuffled_label_ic_small` (`tests/test_models.py`) uses the analogous
logic at the level of a single specialist fit (not the full pipeline): $n_{\text{reps}}=16$
cheaper single-specialist fits, whose *pooled* mean IC is checked against a fixed
$|{\cdot}| < 0.02$ bound — fixed there because pooling 16 reps' predictions before averaging IC
(rather than averaging 16 reps' own mean ICs and using their spread) already gives a tight enough
null check without needing an adaptive bound.
