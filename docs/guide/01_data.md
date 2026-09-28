# Chapter 1 — Data: from raw parquet files to the model-ready panel

This chapter covers `src/data.py`, `src/config.py`, and the two raw files in `data/`. Everything
here is either quoted from the code, quoted from `readme.md` / the hackathon brief, or explicitly
labelled **(measured)** — a number produced by actually running the pipeline against the real
files in this repo (`data/chars_final_with_names.parquet`, `data/8k_20150101_20260831_identified.parquet`,
`outputs/cache/panel.parquet`) on 2026-09-28. Code citations use `file.py:function` form.

---

## 1. The raw files

Two parquet files ship in `data/`, tied together by `readme.md` (the dataset guide) and pinned by
path in `src/config.py:CHARS_PATH` / `config.py:FILINGS_PATH`.

### 1.1 `chars_final_with_names.parquet` — the numeric panel

- **Observation unit:** one security × calendar month.
- **Shape (measured):** `529,082` rows × `198` columns — matches `readme.md` section 1 exactly.
- **Keys:** unique on `(permno, eom)` and also on `(gvkey, iid, eom)` (readme.md section 2).
  `permno` is the CRSP security id and is the key `data.py` uses throughout.
- **Coverage (measured):** `eom` runs `2015-01-31` → `2026-08-31`; `6,561` distinct `permno`.
- **147 characteristics vs 51 auxiliary columns.** The panel has exactly 198 columns because
  `readme.md` states the release keeps "147 selected characteristics plus 51 identifiers and
  auxiliary columns" (147 + 51 = 198). The 147 names live in `data/factor_char_list.csv`'s
  `variable` column, loaded by `config.py:load_char_list()`, which asserts `len(names) == 147`.
  They span valuation, profitability, investment, past-return/momentum, risk/liquidity and
  composite-quality measures (`readme.md` section 4) — e.g. `be_me`, `niq_be`, `at_gr1`, `ret_12_1`,
  `beta_60m`, `ami_126d`, `f_score`, `qmj`. Two of the 147 — `prc` and `market_equity` — are *also*
  identifier-like quantities, but `readme.md` is explicit they are "not counted a second time as
  auxiliary variables"; they sit only in the 147.
  The other 51 columns (listed in full in `readme.md` section 4) are identifiers (`permno`,
  `permco`, `gvkey`, `iid`, `excntry`, …), timing (`date`, `eom`), prices/volume (`prc_local`,
  `prc_high`, `prc_low`, `dolvol`, `tvol`), market cap (`me`, `me_company`, `me_lag1`), returns
  (`ret`, `ret_local`, `ret_exc`, `ret_exc_lead1m`, `ret_exc_wins`), classifications (`gics`, `sic`,
  `naics`, `ff49`, `size_grp`), and **label-provenance columns**: `ticker`, `company_name`,
  `ticker_name_reference_date`, `ticker_name_source`, `ticker_name_status`. These last three record
  *how* a ticker/company-name pair was assigned to a row — not a signal, but audit metadata for
  `portfolio.py`'s label attachment logic (chapter on `portfolio.py`).
  `ticker_name_status` breaks down (measured, full raw file):
  | status | rows (measured) |
  |---|---:|
  | `verified_in_observation_month` | 498,706 |
  | `verified_in_filing_on_or_before_observation_date` | 15,731 |
  | `no_verified_label_for_observation_period` | 14,645 |
  These match `readme.md`'s published counts exactly (514,437 labelled + 14,645 unlabelled).
- **Return-source provenance.** `source_crsp` = 1 (CRSP) or 0 (Compustat). Measured over the whole
  raw file: `500,623` rows source_crsp=1, `28,459` rows source_crsp=0. Restricting to `eom >=
  2026-01-31` (measured `28,453` rows), **every single one** has `source_crsp == 0`: all of 2026
  is Compustat-sourced. This matches `readme.md`'s "All 2026 rows use Compustat" and is elaborated
  in §9 below (return-quality caveat).

### 1.2 `8k_20150101_20260831_identified.parquet` — the filing archive

- **Observation unit:** one retained Form 8-K (or 8-K/A) filing linked to a security.
- **Shape:** `373,139` rows × `94` columns (`readme.md` section 1); `373,054` original 8-Ks + `85`
  amendments. `3,687` distinct `permno`, all of which also appear in the characteristics panel.
- **Keys:** unique on `document_id`, and on `(permno, filing_date, text_sha256)`.
- Not owned by this chapter's code (`src/text.py` consumes it), but `data.py` never touches it —
  `data.py` is characteristics/market data only.

---

## 2. Time conventions — the single most important thing

Quoting `docs/SPEC.md` section 2 verbatim, because getting this wrong is the single most common
way a hackathon submission leaks the future:

> `eom` = formation month t (month-end, datetime64[ns]). Every feature on row `eom=t` uses only
> information available by the end of month t. `target_month` = month-end of t+1. `stock_exret` =
> `ret_exc_lead1m` (excess return realized during target_month). It is already led: never shift it
> again; never use it (or anything derived from it) as a feature.

Concretely, in `src/data.py:_build`:

```python
target_month = df['eom'] + pd.offsets.MonthEnd(1)
stock_exret = df['ret_exc_lead1m']
```

`ret_exc_lead1m` is **already** a one-month-forward excess return as delivered in the raw file —
the provider computed `ret_exc(t+1)` and attached it to row `t`. The pipeline's only job is to
copy it into `stock_exret` (a rename, not a shift) and compute `target_month = eom + 1 month`
purely as a bookkeeping label. **Why never shift again:** `ret_exc_lead1m` is not a same-month
return that needs to be pushed forward — it has already had that done to it upstream. Applying
`.shift(-1)` (or any other lag/lead operator) to it a second time would silently swap in the
excess return from month t+2 while still labelling the row "month t", which both breaks the
train/test time fence (a model could then see information from further in the future than
intended) and would be nearly undetectable in aggregate statistics (it's still a valid-looking
number, just for the wrong month). `readme.md` says exactly this: "It has already been led; do
not shift it again."

**Trade timing.** "Trade at start of t+1" means: a position sized using characteristics known as
of `eom = t` is entered at the start of calendar month t+1 (i.e., effectively at `t`'s month-end
close) and its return outcome is `stock_exret` — the return earned *during* `target_month = t+1`.
`docs/SPEC.md` section 2: "Positions are formed at end of t, held during target_month." This is
also why `docs/SPEC.md` never lets `eom` itself be treated as a return-realization date: `eom` is
strictly a formation/decision date, `target_month` is strictly a holding/realization date.

### Worked example: PERMNO 12490 (IBM)

Measured directly from `data.build_panel()`:

| `permno` | `eom` | `target_month` | `stock_exret` | `date` |
|---:|---|---|---:|---|
| 12490 | 2015-01-31 | 2015-02-28 | 0.063726 | 2015-01-30 |
| 12490 | 2015-02-28 | 2015-03-31 | -0.008884 | 2015-02-27 |
| … | … | … | … | … |
| 12490 | 2026-06-30 | 2026-07-31 | -0.207587 | 2026-06-30 |
| 12490 | 2026-07-31 | 2026-08-31 | 0.058020 | 2026-07-31 |
| 12490 | 2026-08-31 | 2026-09-30 | **NaN** | 2026-08-28 |

IBM has **140** panel rows (measured), consistent with the readme's example rows (its first row,
`eom=2015-01-31`, `me≈151,930.21`M, matches `readme.md` section 6's printed IBM example exactly,
confirming `data.py` reproduces the source values unchanged). Reading the first row: at formation
month `t = 2015-01-31` (using information available through end-of-January-2015), the model would
form a position in IBM; that position is *held through* `target_month = 2015-02-28`, earning
`stock_exret = 0.063726` (6.37% excess return) realized during February 2015. The final row shows
the terminal-month behaviour: `eom = 2026-08-31` (the raw panel's last observed month for IBM) has
no row for `eom = 2026-09-30` to supply a forward return from, so `stock_exret` is `NaN` — "the
missing future return in the terminal observation must not be replaced with zero" (`readme.md`
section 6; also `docs/SPEC.md` A4, §9 below).

This maps to "trade at start of t+1" as: characteristics observed as of the January-2015 month-end
close (`eom=2015-01-31`) determine the position that is effectively established for trading in
February 2015 (`target_month=2015-02-28`) and whose P&L is `stock_exret`.

**Train/valid/test assignment is always keyed by `target_month`**, never by `eom` (`docs/SPEC.md`
section 2): "For test year Y in 2021..2026: train = target months 2015-02..(Y-3)-12; valid =
(Y-2)-01..(Y-1)-12; test = Y-01..Y-12." The first available `target_month` is `2015-02-28`
(`config.py:FIRST_TARGET`) because `eom` starts at `2015-01-31` and there is no `2014-12-31` row to
supply a December 2014 formation month. **Universe membership is decided from `eom`-month
information only and never from whether `stock_exret` exists** — training/validation rows
additionally *require* a non-null label (you need a label to fit against), but test rows do not,
and the *universe itself* (§3 below) is computed before labels are even considered.

---

## 3. Universe definition

`src/data.py:universe_mask`:

```python
def universe_mask(raw: pd.DataFrame) -> pd.Series:
    thresh = raw.groupby('eom')['me'].transform(lambda s: s.quantile(config.ME_CUTOFF_PCTILE))
    return ((raw['prc'] >= config.MIN_PRICE) & (raw['me'] >= thresh)).fillna(False)
```

With `config.MIN_PRICE = 5.0` and `config.ME_CUTOFF_PCTILE = 0.20`, a raw stock-month row is in the
investable universe iff:

$$
\text{prc}_{i,t} \ge \$5 \quad \text{AND} \quad \text{me}_{i,t} \ge Q_{0.20}\big(\{\text{me}_{j,t} : j \in \text{all raw rows at } t\}\big)
$$

i.e. price at least $5 **and** market-equity at least the 20th percentile of market-equity *among
all raw rows sharing the same `eom`* (not just already-filtered rows — the percentile threshold is
computed once per month from the full raw cross-section, then applied). Any row with a null `prc`
or `me` (so the comparison is undefined) is excluded via `.fillna(False)`.

**Why these two filters.** This is the standard "investable universe" screen used across the
empirical asset-pricing literature (and implicitly assumed by the brief's mandate that the
strategy be tradeable): a sub-$5 price excludes penny stocks with disproportionate bid-ask spreads,
non-standard tick sizes and thin/manipulable liquidity; the market-equity bottom-quintile cut
excludes micro-caps that are difficult to trade at meaningful size without moving the price and
that dominate raw characteristic panels in *count* while contributing little in dollar terms. Both
are computed from **month-t information only** — `prc` and `me` are contemporaneous, not future,
quantities.

**Why target availability never defines the universe.** `universe_mask` takes a `raw` frame and
never references `ret_exc_lead1m` at all — the function signature/body simply has no access to it
being relevant (it's not read for this computation). Two invariance tests enforce this explicitly:
`tests/test_data.py:test_universe_mask_ignores_ret_exc_lead1m` and
`tests/test_integrity.py:test_universe_invariance_to_nan_ret_exc_lead1m_real`, both of which set
`ret_exc_lead1m` to `NaN` (or random noise) for a subset of rows and assert the universe mask is
bit-for-bit unchanged. The reasoning: if you let "does this stock have a valid next-month return"
determine whether it's *in the investable universe this month*, you implicitly condition the
universe on the stock still existing and being liquid enough to have a recorded return next month
— a subtle look-ahead/survivorship channel (a stock about to delist non-randomly tends to have
worse forward-looking characteristics). `docs/SPEC.md` section 2 states this as a hard rule:
"Universe membership is decided from month-t information only and never from whether `stock_exret`
exists."

**Measured universe sizes.** Applying `universe_mask` and building the panel (`build_panel()`):
raw rows `529,082` → universe rows `380,865` (**measured**, ≈72.0% pass rate), spanning `140`
distinct `eom` months, `5,756` distinct `permno` (down from 6,561 raw — some names never clear the
$5/20th-percentile bar in any month). Per-month universe size (measured): minimum `2,443`, maximum
`3,232`, mean `2,720`. First month `2015-01-31`: `2,764` names; last month `2026-08-31`: `2,459`
names; `2021-01-31` (the formation month for the first test-period *holding* month, 2021-02 — the
first test formation month itself is `2020-12-31`, whose holding month is 2021-01): `2,818` names,
versus `3,722` raw rows that same month (universe keeps ≈75.7% of raw names in that month).

---

## 4. Feature construction

### 4.1 Within-month rank to [-1, 1]

`src/data.py:_rank_within_eom`:

```python
def _rank_within_eom(df, cols):
    g = df.groupby('eom')[cols]
    ranked = g.rank(method='average')
    counts = g.transform('count')
    scaled = 2 * (ranked - 1) / (counts - 1) - 1
    return scaled.fillna(0.0)
```

For each of the 147 characteristics, within each `eom` cross-section, every **non-null** value is
converted to a percentile-style rank $r_i \in \{1, \dots, n\}$ (pandas `rank(method='average')`:
tied values receive the *average* of the ranks they would occupy — e.g. two values tied for
ranks 2 and 3 both get rank 2.5), where $n$ is the count of non-null values that month for that
column. The rank is then mapped to $[-1, 1]$ by:

$$
\text{scaled}_i = \frac{2(r_i - 1)}{n - 1} - 1
$$

so the smallest value in the cross-section maps to exactly $-1$ and the largest to exactly $+1$,
linearly in between. **Ties**: averaged before scaling, so tied observations get identical scaled
values (as verified by `tests/test_data.py:test_rank_mapping_range_ties_nan`, which hand-computes
`expected_2 = 2*(2.5-1)/3 - 1` for two tied values). **NaN handling**: a null input value has no
rank (`pandas.rank` leaves it `NaN`), and the final `.fillna(0.0)` maps it to exactly `0.0` — the
*midpoint* of the $[-1,1]$ range, i.e. "characteristically neutral" rather than an extreme. **The
$n=1$ case**: when only one non-null observation exists in a month for a column, the division
$\frac{2(r-1)}{n-1}$ has a zero denominator ($n-1=0$); pandas silently produces `NaN` for this
degenerate division regardless of the numerator, and the same `.fillna(0.0)` catches it too — so a
lone observation also lands on `0.0`, not on some arbitrary $\pm1$. This is confirmed by
`tests/test_data.py:test_rank_mapping_n_equals_1_is_zero`.

**Why ranks, not raw or z-scored values.** Cross-sectional rank-based signals are the standard
choice in this literature (JKP-style panels) for two robustness reasons: (1) they are invariant to
the *units and distributional shape* of each raw characteristic — a fat-tailed ratio like
`dolvol_126d` and a bounded ratio like `f_score` end up on exactly the same $[-1,1]$ scale, so a
single model (Ridge/LightGBM/etc.) doesn't need per-feature scale tuning; (2) they are robust to
outliers — a single absurd raw value (e.g. a data error or an extreme M&A-driven ratio) can only
ever occupy one rank position and move to $\pm1$ at most, whereas a z-score would let it dominate
the loss function. Both properties matter especially given the "2026 rows are all Compustat-derived"
caveat (§9) — Compustat-vintage return/accounting data can carry unusual values, and rank-transforming
each column, each month, independently neutralizes that.

### 4.2 Missing-value flags

`src/data.py:_build`:

```python
cutoff_rows = df.loc[df['eom'] <= config.MISS_FLAG_CUTOFF, chars]
miss_rate = cutoff_rows.isna().mean()
miss_chars = miss_rate[miss_rate > config.MISS_FLAG_RATE].index.tolist()
miss_flags = {f'miss_{c}': df[c].isna().astype('int8') for c in miss_chars}
```

The set of characteristics that get a `miss_<char>` flag is *selected once*, using **only
universe rows with `eom <= config.MISS_FLAG_CUTOFF` (2018-12-31)**, as the rows whose null rate
in that window exceeds `config.MISS_FLAG_RATE` (0.20, i.e. 20%). The resulting flag list is then
*applied* (computed as an `int8` "was this raw value null" indicator) to **all** universe rows,
including post-2018 ones. Measured against the real universe rows (rows with `eom <= 2018-12-31`
only, as the selection step does):

| char | measured missing rate (eom ≤ 2018-12-31) |
|---|---:|
| `rd5_at` | 58.5% |
| `rd_sale` | 50.7% |
| `rd_me` | 48.8% |
| `dsale_dinv` | 29.0% |
| `inv_gr1` | 27.8% |
| `intrinsic_value` | 24.4% |
| `pi_nix` | 24.2% |
| `seas_2_5na` | 24.1% |
| `seas_2_5an` | 24.0% |
| `ret_60_12` | 23.8% |
| `ope_bel1` | 23.6% |
| `debt_gr3` | 22.2% |
| `ope_be` | 21.7% |
| `aliq_mat` | 21.6% |
| `dsale_dsga` | 21.1% |
| `f_score` | 20.6% |

These 16 exceed the 20% threshold (measured — matches exactly the 16 `miss_` columns present in
`outputs/cache/panel.parquet`: `miss_aliq_mat, miss_debt_gr3, miss_dsale_dinv, miss_dsale_dsga,
miss_f_score, miss_intrinsic_value, miss_inv_gr1, miss_ope_be, miss_ope_bel1, miss_pi_nix,
miss_rd_me, miss_rd_sale, miss_rd5_at, miss_ret_60_12, miss_seas_2_5an, miss_seas_2_5na`). The next
runners-up (`ni_ivol`, `ni_ar1`, `earnings_variability`, `qmj`, all ≈18.5–18.8%) fall just short of
20% and get no flag.

**Why the cutoff avoids look-ahead.** The choice of *which* 16 characteristics deserve a
missingness flag is itself a modelling decision (like choosing which features to include), and
`docs/SPEC.md` section 2's expanding-window schedule starts its *first* validation window at
2019-2020 (test year 2021: valid = 2019-01..2020-12) and its first test year at 2021. By computing
the missing-rate statistic **only** from data dated on or before 2018-12-31 — strictly before any
validation or test window ever used in the whole 2021–2026 schedule — the selection of *which*
characteristics get a flag cannot have been influenced by patterns in data the model is later
evaluated against, for *any* of the six test years. This mirrors the general "fit learned
preprocessing on admissible training information" instruction in the brief (page 13) and is
verified not to leak by `tests/test_data.py:test_miss_flags_use_only_cutoff_rows`, which
constructs a characteristic that is missing *only* in months after the cutoff and asserts no flag
is created for it.

### 4.3 The `prc` / `dolvol_126d` raw-vs-ranked name collision

`prc` and `dolvol_126d` are each *two things at once* in this pipeline: (a) one of the 147 selected
characteristics (so it gets rank-transformed to $[-1,1]$ like every other characteristic), and (b)
a raw-valued auxiliary quantity that other modules need untransformed (e.g. `portfolio.py`'s
short-book liquidity checks want actual dollar volume, not a rank). `src/data.py:_build` resolves
this explicitly (its own inline comment, quoted verbatim):

> NOTE deviation from literal SPEC wording: 'prc' and 'dolvol_126d' are both feature-char names
> (ranked above) and requested as raw aux columns under the same name, which collide. We keep the
> ranked feature under the standard name (needed by models.py) and expose the raw value under
> `<name>_raw` instead. 'me' has no such collision and stays raw.

Concretely: the panel's `prc` and `dolvol_126d` columns are the **rank-transformed [-1,1] feature
values** (used by `models.py`), while `prc_raw` and `dolvol_126d_raw` carry the untouched dollar
values (measured: `prc_raw` mean ≈ $64.70, `dolvol_126d_raw` mean ≈ $105.6M). `beta_60m` has a
similar raw-copy need (it feeds the `beta` shrinkage formula in §5) but no naming collision, since
the shrunk aux column is named `beta`, not `beta_60m`; `_build` still snapshots `beta_raw =
df['beta_60m'].copy()` before rank-transforming `beta_60m` in place, to compute the shrinkage from
the untransformed value. `tests/test_integrity.py:test_feature_hygiene_real` asserts no `*_raw`
column ever leaks into `feature_columns()` — they are aux-only, never model inputs.

---

## 5. Auxiliary columns

Computed in `src/data.py:_build`, none of these are model features (`feature_columns()` excludes
them):

- **`gics2`** — first 2 characters of the `gics` industry code (sector-level granularity), or
  the string `'NA'` when `gics` is null:
  ```python
  gics2 = np.where(df['gics'].isna(), 'NA', df['gics'].astype(str).str.slice(0, 2))
  ```
  Measured: `12` distinct sector codes among universe rows, plus `'NA'` for `12,556` rows
  (`gics` missing). Used by `portfolio.py` for sector-neutrality constraints (`SECTOR_TOL`).

- **`beta`** — a shrunk version of `beta_60m` (the raw 60-month rolling market beta, one of the
  147 characteristics):

  $$
  \text{beta}_i =
  \begin{cases}
  1.0 & \text{if } \text{beta\_60m}_i \text{ is missing} \\
  (1 - \text{BETA\_SHRINK}) \cdot \text{beta\_60m}_i + \text{BETA\_SHRINK} \cdot 1.0 & \text{otherwise}
  \end{cases}
  $$

  With `config.BETA_SHRINK = 0.33`, this is $0.67 \times \text{beta\_60m} + 0.33 \times 1.0$ for
  observed betas, and exactly `1.0` when `beta_60m` is missing. Code:
  ```python
  beta = np.where(beta_raw.isna(), 1.0,
                  (1 - config.BETA_SHRINK) * beta_raw + config.BETA_SHRINK * 1.0)
  ```
  **Why shrink toward 1.0.** This is a Blume/Vasicek-style shrinkage: a noisily-estimated
  historical beta (60 months of returns is a short, high-variance window, especially for younger
  or thinly-traded names) is pulled partway toward the market's own beta of 1.0, which is the
  Bayesian-flavoured, cross-sectionally-informed prior that a typical stock's *true* beta is close
  to the market average, and that extreme estimated betas are disproportionately estimation noise
  rather than true risk differences (the classic Blume 1971 / Vasicek 1973 result). $0.67 \cdot
  \beta + 0.33$ is exactly the classic Blume (1971) adjustment ($\beta_{adj} = 2/3 \cdot \beta +
  1/3$). Missing `beta_60m` gets
  the *full* shrinkage target (1.0) rather than being left null, so downstream consumers (the
  optimizer's `|beta @ w| <= BETA_TOL` constraint in `portfolio.py`) always have a usable number.
  Measured: of `380,865` universe rows, `64,506` (16.9%) have `beta_60m` missing and thus `beta ==
  1.0` exactly; overall `beta` has mean ≈1.095, std ≈0.386, range roughly [-2.16, 7.16].

- **`size_z`** — within-`eom` z-score of `log(me)`:
  ```python
  log_me = np.log(df['me'])
  size_z = log_me.groupby(df['eom']).transform(lambda s: (s - s.mean()) / s.std())
  ```
  i.e. $\text{size\_z}_{i,t} = \dfrac{\ln(\text{me}_{i,t}) - \overline{\ln(\text{me}_t)}}{\sigma(\ln(\text{me}_t))}$,
  a standard log-size standardization used by `portfolio.py`'s `|size_z @ w| <= SIZE_TOL`
  neutrality constraint. Measured: mean ≈0 (by construction), std ≈1.0.

- **Other raw pass-throughs**: `me`, `size_grp`, `ticker`, `company_name` are copied unchanged
  from the raw panel (not rank-transformed, not recomputed) alongside `prc_raw`, `dolvol_126d_raw`
  from §4.3.

---

## 6. `market_state()`

`src/data.py:market_state()` builds one row per `eom` from raw (not universe-filtered, for
`mkt_ret`) and universe-filtered (for `disp`/`ivol`) data, using data `<= t` only — enforced by
`tests/test_data.py:test_market_state_truncation_invariance` and
`tests/test_integrity.py:test_truncation_invariance_market_state_real`, both of which check that
rebuilding `market_state()` from data truncated at some month `T` reproduces row `T` exactly.

- **`mkt_ret`** — a market-cap-weighted mean return across *all* raw panel rows that month (not
  universe-restricted), weighted by **prior-month** market equity `me_lag1` (falling back to the
  current-month `me` when `me_lag1` is missing — measured: only ≈0.55% of raw rows have `me_lag1`
  missing):
  ```python
  weight = raw['me_lag1'].fillna(raw['me']).where(raw['ret'].notna())
  w_ret = raw['ret'] * weight
  mkt_ret = w_ret.groupby(raw['eom']).sum() / weight.groupby(raw['eom']).sum()
  ```
  $$
  \text{mkt\_ret}_t = \frac{\sum_i w_{i,t}\, \text{ret}_{i,t}}{\sum_i w_{i,t}}, \quad w_{i,t} = \text{me\_lag1}_{i,t} \text{ (or me}_{i,t}\text{ if missing)}
  $$
  Using `me_lag1` rather than contemporaneous `me` for the weight avoids weighting by a market cap
  that itself already reflects the current month's return (a small look-ahead/mechanical-correlation
  issue when weight and return are measured over the same period).

- **`mkt_ret12`** — trailing 12-month **compounded** market return, minimum 6 months of history:
  ```python
  state['mkt_ret12'] = state['mkt_ret'].rolling(12, min_periods=6).apply(lambda x: np.prod(1+x)-1)
  ```
  $$
  \text{mkt\_ret12}_t = \prod_{k=0}^{11} (1 + \text{mkt\_ret}_{t-k}) - 1
  $$

- **`mkt_vol12`** — trailing 12-month standard deviation of `mkt_ret`, minimum 6 months:
  `state['mkt_ret'].rolling(12, min_periods=6).std()`.

- **`disp`** — cross-sectional dispersion (std) of *universe* returns that month, computed after
  clipping each stock's `ret` within-month to the [1st, 99th] percentile:
  ```python
  clipped_ret = uni.groupby('eom')['ret'].transform(lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))
  disp = clipped_ret.groupby(uni['eom']).std()
  ```
  **Why the 1%/99% clip.** The code comment states the reason directly: "raw ret has rare outliers
  up to +3000% that would otherwise dominate an unclipped std" — a handful of extreme
  data-artifact or corporate-action-driven monthly returns would otherwise make `disp` a noisy
  statistic dominated by one or two names rather than a genuine cross-sectional-dispersion signal.

- **`ivol`** — simple (unweighted) mean of `ivol_capm_21d` (one of the 147 characteristics, a
  21-day CAPM idiosyncratic volatility) across universe rows that month.

**Measured** (full history, `140` months): `mkt_ret` mean ≈1.16%/mo, std ≈4.49%/mo, minimum
-13.24% in 2020-03; `disp` mean ≈11.7%, ranging 7.6%–19.5%; `ivol` mean ≈2.27%. The 2020-03 row
(measured) shows `mkt_ret=-0.1324`, `disp=0.1679` (dispersion spikes in the COVID crash month, as
expected), `mkt_ret12=-0.0867`, `mkt_vol12=0.0595` — all directionally sensible.

**Where each is used.** `config.py:STATE_VARS = ['mkt_vol12', 'disp']` — only these two of the five
`market_state()` columns feed the gate's regime-interaction ablation (`docs/SPEC.md` section 5:
"Gate design = [Z_k] + [Z_k * s_j] for s_j in config.STATE_VARS"). `mkt_ret` and `mkt_ret12` are
computed and available but not wired into the gate's interaction terms; `ivol` likewise is
descriptive/available but not one of the two chosen state variables. (`docs/research_log.md`'s
A12 entry separately explains that the gate itself was ultimately demoted from headline to
ablation status on validation-window evidence — a `models.py`-chapter topic, not a `data.py` one.)

---

## 7. `load_market()`

`src/data.py:load_market()` provides two external time series, both cached under `data/external/`,
plus derived quantities:

- **TB3MS** (3-Month Treasury Bill, Secondary Market Rate) from FRED:
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS`. **Units:** annualized percent (e.g.
  `4.20` means 4.20%/year) — matching the brief's own instruction (page 8) to divide by 100 and by
  12 for a monthly decimal rate. **Month alignment:** FRED publishes TB3MS dated to the
  *first* of each month; `_download_market_data` remaps each date to that month's month-end via
  `.dt.to_period('M').dt.to_timestamp('M')` so it aligns with the panel's `eom` convention.
- **S&P 500 total return**: `yfinance` ticker `^SP500TR`, monthly interval, `Close` prices,
  `pct_change()` (i.e. month-over-month simple return, decimal). `^SP500TR` is the *total*-return
  index (dividends reinvested), which is why it's preferred over a bare price index — the fallback
  path (FRED `SP500`, a *price-only* index) is used only if `yfinance` errors, and is explicitly
  flagged `sp500_source = 'price_only'` with a written caveat in `SOURCES.md` that "`sp500_ret`
  therefore understates true total return." As of this write-up, the primary source succeeded:
  measured `data/external/sp500.csv`'s `sp500_source` column reads `sp500tr_total_return`
  throughout. **Sanity values (measured):** 2022 compounded `sp500_ret` = **-18.11%**
  (`tests/test_data.py:test_real_sp500_total_return_sanity` expects -18.1% ± 0.5pp — the S&P 500
  total return was indeed roughly -18% in calendar 2022); 2020-03 `sp500_ret` = **-12.35%** (test
  expects -12.4% ± 0.5pp, the well-known COVID crash month); `tb3ms` at 2023-07 = **5.25%** (test
  expects 5.0–5.5%, consistent with Fed funds near its 2023 peak); `tb3ms` at 2021-01 = **0.08%**
  (test expects < 0.2%, consistent with near-zero rates in early 2021).
- **`rf_m = tb3ms / 1200`** — converts the annualized percent to a monthly decimal rate
  ($\div 100$ for percent→decimal, $\div 12$ for annual→monthly, combined as $\div 1200$).
- **`sp500_exret = sp500_ret - rf_m`** — S&P 500's own monthly excess return, used later for the
  market-neutrality regression (`evaluate.py:alpha_beta`).
- **Partial-month dropping.** `load_market()` always drops any row for the current, not-yet-closed
  calendar month:
  ```python
  df = df.loc[df.index < _current_month_end()]
  ```
  where `_current_month_end()` is the month-end of `pd.Timestamp.today()`. This guards against
  FRED or `yfinance` returning a partial/incomplete bar for an in-progress month (whether or not a
  cached csv happens to already include one) — such a bar is not a true month-end observation and
  would corrupt any month-end-keyed join. As of 2026-09-28 (today), this drops any would-be
  2026-09 row; the cached files (measured, `data/external/tb3ms.csv` / `sp500.csv`) end at
  `2026-08-31`, so nothing is currently being dropped in practice, but the guard applies regardless
  of what's on disk.
- **Caching**: on first call, if `tb3ms.csv` and `sp500.csv` don't already exist under
  `config.EXT_DIR`, `_download_market_data()` fetches both series once, writes them to those CSVs,
  and writes a `SOURCES.md` documenting the URLs/units/source flag; every later call reads only
  the cached CSVs (no re-download). Coverage requirement (`docs/SPEC.md` section 3): "Must cover
  2020-12..2026-08 at least" — measured, `load_market()` returns `141` rows spanning
  `2014-12-31`→`2026-08-31`, comfortably covering the required window, with no NaNs in
  `tb3ms`/`rf_m`/`sp500_ret`/`sp500_exret` over 2020-12..2026-08 (verified by
  `tests/test_data.py:test_real_load_market_coverage`).

---

## 8. Caching behavior and its staleness hazard

`src/data.py:build_panel()`:

```python
def build_panel() -> pd.DataFrame:
    cache_path = config.CACHE_DIR / 'panel.parquet'
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    ...
    panel = _build(raw)
    panel.to_parquet(cache_path)
    return panel
```

The cache check is **existence-only** — it never compares the cache's modification time (or a
content hash) against `config.CHARS_PATH`'s modification time, and never re-reads `factor_char_list.csv`
or re-checks `config.MISS_FLAG_CUTOFF`/`MIN_PRICE`/`ME_CUTOFF_PCTILE`/`BETA_SHRINK` once
`outputs/cache/panel.parquet` exists on disk. **This is the staleness hazard**: if the raw
`chars_final_with_names.parquet` is replaced (e.g. a corrected/updated data drop from the
organizers), or if `config.py`'s universe/rank/flag/shrinkage constants are edited, `build_panel()`
will keep silently returning the *old* cached panel until someone manually deletes
`outputs/cache/panel.parquet`. Measured file timestamps in this repo: the raw file
`data/chars_final_with_names.parquet` was last modified 2026-09-17, and the cached
`outputs/cache/panel.parquet` was last modified 2026-09-27 — i.e. currently the cache postdates the
raw file and is not stale, but nothing in the code enforces that relationship going forward. The
`_build()` helper (the pure-function core, separated from the on-disk cache by design, per its own
docstring: "Split out from build_panel so it is callable without touching the on-disk cache") is
what the truncation-invariance tests use to sidestep this hazard when testing against modified
inputs. `market_state()` and `load_market()` have their own caching behaviors: `market_state()` is
**not** cached to disk at all (recomputed from `load_chars()` on every call — cheap enough not to
need it); `load_market()`'s two CSV caches (§7) share the same existence-only staleness hazard —
`tb3ms.csv`/`sp500.csv`, once written, are never refreshed automatically even though the world
(actual TB3MS/S&P levels) moves forward every month.

---

## 9. Data caveats

- **2026 Compustat-sourced returns.** Measured: every one of the `28,453` raw rows with `eom >=
  2026-01-31` has `source_crsp == 0` (Compustat), versus `source_crsp == 1` (CRSP) for essentially
  all prior history. `readme.md` flags this directly ("`source_crsp`: Return-data source: `1` =
  CRSP, `0` = Compustat. All 2026 rows use Compustat.") and separately notes `ret_exc_wins`
  clips *Compustat* values at the pipeline's period-specific 0.1%/99.9% cutoffs while leaving CRSP
  values unchanged — i.e. the two return sources are not perfectly homogeneous in their outlier
  treatment, a fact worth remembering when interpreting any 2026 test-period result. `data.py`
  itself does not special-case `source_crsp` (it neither excludes Compustat rows nor uses
  `ret_exc_wins` in place of `ret`/`ret_exc_lead1m`), so this caveat is purely informational for
  interpreting results, not something `data.py`'s code branches on.

- **Missing next-month returns.** `docs/SPEC.md` A4: "Missing next-month returns (0.47% of
  test-universe stock-months, all true exits) stay 0 in the headline." Measured directly against
  `outputs/cache/panel.parquet`, restricting to rows whose `target_month` falls in the competition
  test window (`config.TEST_START=2021-01-31` .. `config.TEST_END=2026-08-31`, `189,448` such
  rows): the `stock_exret` null rate is **0.4671%** — matching the documented 0.47% figure exactly.
  (Over the *entire* panel, not just the test window, the null rate is measured at 1.06% — higher,
  because it also includes the panel's final month, 2026-08-31, whose forward return cannot exist
  at all yet.) These are all cases where a `permno` simply has no row at `eom = target_month` in
  the raw source (a true delisting/exit, not a data error), so no `ret_exc(t+1)` was ever available
  to lead into `ret_exc_lead1m`. `portfolio.py`'s backtest treats a missing realized return as
  `0` for accounting purposes in the headline result (tracked separately via
  `missing_ret_weight`), with an adverse -30%/+30% sensitivity reported as a robustness check —
  the `data.py`-level fact this rests on is simply that `stock_exret` is `NaN`, never silently
  zero-filled, for these rows (`docs/SPEC.md` A4: "Labels are never zero-filled for training").

- **Labels missing for some rows.** The Opus audit found that every missing test-window label
  (885 stock-months, 0.467% — the figure in the bullet above) occurs at a permno's *last* panel
  row, i.e. a true exit: there is no mid-panel case where `stock_exret` is null while the permno
  continues to appear in later rows. `data.py` never manufactures a label and always represents
  "no usable next-month outcome" as `NaN`, letting `models.py`'s split logic (`docs/SPEC.md`
  section 2: "Training/validation rows additionally require non-null `stock_exret`") filter
  appropriately rather than `data.py` silently dropping or imputing rows itself.

---

## 10. Data tests, and what each proves

### `tests/test_data.py` (fast, synthetic unless `@pytest.mark.slow`)

| Test | What it proves |
|---|---|
| `test_rank_mapping_range_ties_nan` | The `_rank_within_eom` formula is exactly $2(r-1)/(n-1)-1$; ties get averaged ranks producing identical scaled values; NaN inputs map to exactly `0.0`; min/max hit exactly $-1$/$+1$. |
| `test_rank_mapping_n_equals_1_is_zero` | The $n=1$ degenerate-division case lands on `0.0`, not an arbitrary extreme. |
| `test_universe_mask_ignores_ret_exc_lead1m` | `universe_mask` output is bit-identical whether `ret_exc_lead1m` is left alone, all-NaN'd, or replaced with random noise — the universe never depends on target availability. |
| `test_target_month_and_stock_exret` | `target_month == eom + 1 month-end` on every row; `stock_exret` exactly equals the raw `ret_exc_lead1m` (never re-shifted), on synthetic data with a known ground truth. |
| `test_feature_columns_excludes_target_derived` | `feature_columns()` never contains `stock_exret`/`ret_exc_lead1m`/`target_month`/`eom`/`date`/`permno`, and has no duplicate names. |
| `test_market_state_truncation_invariance` | Rebuilding `market_state()` from data truncated at month `T` reproduces row `T` exactly — proves no rolling window peeks past `T`. |
| `test_miss_flags_use_only_cutoff_rows` | A characteristic missing *only* in months after `MISS_FLAG_CUTOFF` gets **no** `miss_` flag — proves flag selection genuinely respects the cutoff and can't be back-doored by post-cutoff missingness. |
| `test_beta_shrink_and_gics2_na` | `beta` shrink formula reproduced exactly on a known raw `beta_60m`; missing `beta_60m` → `beta == 1.0` exactly; missing `gics` → `gics2 == 'NA'`; present `gics` → correct 2-char slice. |
| `test_real_panel_unique_and_ranks` *(slow)* | On the real panel: `(permno, eom)` is unique (no duplicate stock-months); every ranked feature value lies in $[-1,1]$; banned target-derived columns are absent from `feature_columns()`. |
| `test_real_stock_exret_matches_next_month_raw` *(slow)* | On a 5,000-row real sample: `stock_exret` at `(permno, eom)` matches the raw `ret_exc` at `(permno, eom=target_month)` in >99.9% of cases where that next-month raw row exists — direct empirical confirmation of the leading/alignment logic against the source data, not just synthetic construction. |
| `test_real_load_market_coverage` *(slow)* | `load_market()` has no gaps over the full 2020-12..2026-08 test window and no NaNs in `tb3ms`/`rf_m`/`sp500_ret`/`sp500_exret` there. |
| `test_real_sp500_total_return_sanity` *(slow)* | Cross-checks cached external data against well-known reference figures (2022 ≈ -18%, 2020-03 ≈ -12.4%, mid-2023 T-bill ≈5.0-5.5%, early-2021 T-bill <0.2%) — catches a wrong series/units/source silently swapped in. |
| `test_real_panel_truncation_invariance` *(slow)* | Same truncation-invariance property as the synthetic version, but on the real panel via the cache-free `_build()` path, with the label at the truncation boundary explicitly wiped — proves features/universe at month `T` are unaffected by both future rows and future labels. |
| `test_real_market_state_truncation_invariance` *(slow)* | Same as above for `market_state()`, on real data. |

### `tests/test_integrity.py` (data-related portions; SPEC section 8 leakage suite)

| Test | What it proves |
|---|---|
| `test_target_alignment_real` | Same next-month-return alignment check as `test_data.py`'s version, run again here as part of the standalone judge-facing integrity suite (module docstring: deliberate overlap so this one file gives full section-8 coverage on its own), plus a *non-sampled* check that `target_month == eom + 1 month-end` holds on **every** panel row. |
| `test_truncation_invariance_panel_real` | Real-data truncation invariance for the whole panel (features, universe membership, and aux columns), again via `_build()` without touching the on-disk cache. |
| `test_truncation_invariance_market_state_real` | Same, for `market_state()`. |
| `test_truncation_invariance_text_real` | (Owned conceptually by `text.py`, but confirms the same no-look-ahead property for text features built from the real 8-K archive truncated at filing_date ≤ T.) |
| `test_feature_hygiene_real` | On the real, full panel + text features: none of `stock_exret`, `ret_exc_lead1m`, `target_month`, `has_filing`, `ticker`, `company_name` ever appear in the feature set; no `*_raw` aux column leaks in; `gics2`/`beta`/`size_z`/`me`/`size_grp`/`date`/`eom`/`permno` are confirmed to be aux/id-only. |
| `test_universe_invariance_to_nan_ret_exc_lead1m_real` | Same universe/target-independence property as `test_data.py`'s version, on the real raw file with a random third of `ret_exc_lead1m` values NaN'd out. |

**Net effect of this suite:** between the fast synthetic tests (which pin down exact formulas with
hand-computed expected values) and the slow real-data tests (which confirm those formulas behave
correctly, and stay leakage-free, against the actual 529,082-row panel and 373,139-row filing
archive), every claim in `docs/SPEC.md` section 3 about `data.py` — the rank formula, the universe
definition, the missing-flag cutoff, the beta shrink, and above all "never look past `eom`" — is
mechanically checked, not just asserted in prose.
