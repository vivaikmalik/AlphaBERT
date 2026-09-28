# Chapter 5 — Evaluation: every statistic, regression, table and chart

`src/evaluate.py` is the last stage of the pipeline. It takes the backtest's raw monthly
output — `returns`, `holdings`, the stock `panel`, and (optionally) the gate coefficients and
market-state series — and turns them into the numbers, tables and charts a reader (or an
investment committee, per the brief's own framing) would actually be handed. Nothing in this
chapter is a claimed *result*: the full 68-month backtest has not been run yet, so every formula
below is verified against the code, not against a number.

All code citations point at `src/evaluate.py` unless another file is named. Section numbers
below map onto `docs/SPEC.md` section 7 ("evaluate.py") and section 10 ("Amendments").

---

## 1. The test window, and what "holding month" means

### 1.1 The 68-month window

The evaluation module hard-codes one restriction, applied everywhere the test period matters:

```python
config.TEST_START = pd.Timestamp('2021-01-31')   # first holding (target) month-end
config.TEST_END   = pd.Timestamp('2026-08-31')    # last holding (target) month-end
```

(`src/config.py:24-25`.) January 2021 to August 2026 inclusive is

$$(2026-2021)\times 12 + (8-1) + 1 = 60 + 8 = 68 \text{ months.}$$

Every table- and chart-producing function receives data that has already been passed through
`_window_mask()` (`src/evaluate.py:117-120`), a boolean mask `TEST_START <= date <= TEST_END`,
and then through `_restrict_window()` (`src/evaluate.py:402-406`):

```python
def _restrict_window(returns):
    out = returns.loc[_window_mask(returns.index)].sort_index()
    missing = pd.date_range(config.TEST_START, config.TEST_END, freq='ME').difference(out.index)
    assert len(missing) == 0, f'evaluate: missing test-window months: {[d.date() for d in missing]}'
    return out
```

This is not just a slice — it is a completeness *assertion*. If a single one of the 68
calendar month-ends is absent from `returns.index`, `run_evaluation()` raises immediately rather
than silently computing statistics over a shorter, unstated window. `run_evaluation()` calls it
on the headline `returns` (`src/evaluate.py:427`), and `ablation_table()` and the sensitivity
branch call it independently on every ablation and on the sensitivity returns
(`src/evaluate.py:277`, `src/evaluate.py:452`) — so a broken ablation signal fails loudly instead
of quietly evaluating over 60 months. `tests/test_evaluate.py::test_restrict_window_asserts_on_missing_month`
exercises exactly this: it deletes one month from a synthetic 68-month index and checks that
`_restrict_window` raises `AssertionError`.

### 1.2 "Holding month" precisely

Per `docs/SPEC.md` section 2 (time conventions), every row in `returns` and `holdings` is keyed
by a **formation month** `eom = t` and a **holding month** (called `month` in `holdings`, and
the DataFrame index in `returns`) equal to `target_month = t + 1` month-end. Positions are
*decided* using only information known at the close of month $t$ (characteristics, text, market
state as of $t$), and then *held* — earning `stock_exret` — during the following calendar month,
$t+1$. So:

- "Formation month" 2020-12-31 → its positions are held, and earn a return, during holding month
  2021-01-31 — the **first** row of the 68-month test window.
- "Formation month" 2026-07-31 → held during 2026-08-31 — the **last** row.

That is why `docs/SPEC.md` section 2 states the test window as holding months 2021-01..2026-08
but formation months 2020-12..2026-07: the two spans are the same 68 months, offset by one, and
`evaluate.py` only ever looks at the holding-month axis (it never sees formation months directly
except through `holdings['eom']`, used purely for point-in-time labeling — see §5).

---

## 2. Every statistic: formula and code location

All of the low-level statistic functions live at the top of `src/evaluate.py` (lines 54–129) and
are then assembled into `performance_table()`. Throughout, `active_ret = total_ret - bench_ret`
and `bench_ret = rf_m + 0.04/12` (from `src/portfolio.py`, per `docs/SPEC.md` section 6), so all
of the "active" and "excess" language below refers to a *monthly* series already computed
upstream — `evaluate.py` never recomputes portfolio returns, only statistics of them.

### 2.1 Average monthly and annualized arithmetic return

`_series_row()` (`src/evaluate.py:133-142`):

$$\overline{r} = \frac{1}{n}\sum_{t=1}^n r_t \qquad (\texttt{avg\_monthly} = \texttt{ret.mean()})$$

$$r_{\text{ann, arith}} = 12\,\overline{r} \qquad (\texttt{ann\_arith} = \texttt{ret.mean()} \times 12)$$

This is the simple (arithmetic) annualization the brief page 21 asks for alongside CAGR — it is
*not* compounded, and will differ from CAGR whenever monthly returns are volatile: the $\times12$
arithmetic annualization ignores compounding, so for a volatile series CAGR is lower, roughly
$\text{CAGR} \approx 12\,\overline{r} - 6\,\sigma^2_{\text{monthly}}$ (i.e. arithmetic minus half
the annualized variance).

### 2.2 CAGR

```python
def cagr(ret):
    ret = ret.dropna()
    return float((1 + ret).prod() ** (12 / len(ret)) - 1)
```
(`src/evaluate.py:72-76`.) This is the standard geometric/compound annual growth rate,
generalized to a fractional number of years via the month count $n$ (68 for the headline
window, $68/12 \approx 5.67$ years):

$$\text{CAGR} = \left(\prod_{t=1}^{n}(1+r_t)\right)^{12/n} - 1$$

`tests/test_evaluate.py::test_cagr_constant_monthly_return` checks the identity directly:
12 months of a constant 1% monthly return should give $\text{CAGR} = 1.01^{12}-1$.

### 2.3 Cumulative return

```python
def cumulative(ret):
    return float((1 + ret.dropna()).prod() - 1)
```
(`src/evaluate.py:79-80`.) $\prod_t (1+r_t) - 1$ — total compounded return over the whole window,
with no annualization. `tests/test_evaluate.py::test_cumulative_and_max_drawdown_known_path` hand
-checks it on `[0.10, -0.50, 1.00]`: $1.10\times0.50\times2.00-1 = 0.10$.

### 2.4 Best / worst month

`_series_row()` uses `r.idxmax()`/`r.idxmin()` for the dates and `r.max()`/`r.min()` for the
values, on the *raw* return series (`total_ret`, `bench_ret`, or `sp500_ret` depending on the
row) — not on active return. Both the date (the pandas Timestamp index) and the value are
recorded, matching the brief's "best month and worst month, with dates" (page 21).

### 2.5 Hit rate

```python
def hit_rate(active):
    return float((active.dropna() > 0).mean())
```
(`src/evaluate.py:96-97`.) Share of months with **positive active return** (not positive total
return) — exactly the brief's definition (page 21: "the share of months with a positive active
return"). It is `NaN` for rows with no active-return series (e.g. the benchmark row itself, see
§2.9).

### 2.6 Calendar-year compounding (incl. 2026 YTD)

```python
def calendar_year_table(returns):
    def compound(s): return (1 + s).prod() - 1
    g = returns.groupby(returns.index.year)
    out = pd.DataFrame({'strategy_gross': g['total_ret'].apply(compound),
                         'strategy_net': g['total_ret_net'].apply(compound),
                         'benchmark': g['bench_ret'].apply(compound),
                         'sp500': g['sp500_ret'].apply(compound)})
    out.index = [f'{y} YTD (Jan–Aug)' if y == 2026 else str(y) for y in out.index]
    return out
```
(`src/evaluate.py:164-173`.) Each calendar year's return is the *compounded* product of that
year's monthly returns, $\prod_{t\in\text{year}}(1+r_t)-1$, not a sum — this is important because
2021, 2022, and 2023-2025 all have differing compounding effects at the ~1-2%/month scale
typical of long-short spreads. 2026 only has 8 months of data (Jan–Aug), so its row is
explicitly labelled `"2026 YTD (Jan–Aug)"` rather than presented as a full-year figure — a
reader who compounds it forward or annualizes it would be badly misled, hence the label.
`tests/test_evaluate.py::test_calendar_year_compounding` verifies the compounding formula against
a hand computation for 2021/2022; `test_calendar_year_2026_labelled_ytd` checks the label string
appears.

### 2.7 Information ratio vs cash + 4%

```python
def ir(active):
    active = active.dropna()
    return _ratio_or_nan(active.mean(), active.std(ddof=1))

IR_FORMULA = ('IR = sqrt(12) * mean(active) / std(active), '
              'active = total_ret - (TB3MS/1200 + 0.04/12)')
```
(`src/evaluate.py:62-64, 45-46`.) With $n=1$ Bessel correction (`ddof=1`, sample std):

$$\text{IR} = \sqrt{12}\,\frac{\overline{\text{active}}}{s_{\text{active}}}, \qquad
\text{active}_t = R^{\text{total}}_{p,t} - \left(\frac{\text{TB3MS}_t}{1200} + \frac{0.04}{12}\right)$$

This is exactly the brief's own definition (page 8: "active_t = R_portfolio,total,t − (TB3MS_t /
100 / 12 + 0.04 / 12)", then "Information Ratio = √12 × mean(active_t) / stdev(active_t)"), and
it is applied to the *active return series itself*, never to a regression intercept divided by
residual volatility. The brief is explicit that the latter — the ratio printed by the provided
`portfolio_analysis_hackathon.py` template — is the wrong quantity for this competition ("its
printed Information Ratio is an alpha-to-residual-volatility ratio, not this competition's
benchmark information ratio," page 15); `evaluate.py` never computes that ratio at all. The
`IR_FORMULA` string is printed to stdout by `run_evaluation()` (`src/evaluate.py:430`) and stored
verbatim in the headline dict (§7) precisely so the formula travels with every number that uses
it.

`performance_table()` computes IR for the strategy (both gross and net legs) using
`returns['active_ret']`/`active_ret_net`, and for the S&P 500 row using `sp500_ret - bench_ret`
as its "active" series (`src/evaluate.py:150`) — i.e. how the index itself would have scored
against the same cash-plus-4% hurdle, which is the natural apples-to-apples comparison the brief
asks the S&P 500 to provide "for context" (page 8, page 20).

### 2.8 Sharpe ratio

```python
def sharpe(ret, rf):
    excess = (ret - rf).dropna()
    return _ratio_or_nan(excess.mean(), excess.std(ddof=1))
```
(`src/evaluate.py:67-69`.) $\text{Sharpe} = \sqrt{12}\,\overline{(r-r_f)}/s_{(r-r_f)}$ — excess
*over the risk-free rate* `rf_m`, per the brief page 8 ("Compute the Sharpe ratio from portfolio
returns in excess of the stated risk-free return; it should not give credit for earning that
risk-free return alone").

**Why the benchmark's own Sharpe is N/A.** For the benchmark row, `ret = bench_ret = rf_m +
0.04/12` and `rf = rf_m`, so `ret - rf` is the *constant* `0.04/12` every month — zero variance,
nonzero mean. `_ratio_or_nan()` (`src/evaluate.py:55-59`) returns `NaN` whenever the denominator
standard deviation is below `_ZERO_STD_TOL = 1e-8`, *regardless of the numerator*:

```python
def _ratio_or_nan(mean, sd):
    if not np.isfinite(sd) or sd < _ZERO_STD_TOL:
        return np.nan
    return float(np.sqrt(12) * mean / sd)
```
This deliberately avoids the alternative bug of returning `+inf` for a strictly-positive,
zero-variance excess series — a ratio with a zero denominator is mathematically undefined, not
infinite, and `_ratio_or_nan` never returns `inf`.
`tests/test_evaluate.py::test_sharpe_benchmark_zero_variance_is_nan_not_inf` plants exactly this
case (`bench_ret - rf_m` constant at `0.04/12`) and asserts both `np.isnan(result)` and
`not np.isinf(result)`. The same mechanism makes the benchmark's own IR (§2.7) and hit rate
`NaN` in `performance_table()` — its `_series_row()` call passes `active=None` explicitly
(`src/evaluate.py:149`), so `ir`/`hit_rate` are never even attempted for that row.
`tests/test_evaluate.py::test_performance_table_benchmark_row_nan_ir_and_hit_rate` checks all
three (`ir`, `hit_rate`, `sharpe`) are `NaN` on the benchmark row. This matches SPEC amendment A5:
"Benchmark Sharpe reported N/A."

### 2.9 Maximum drawdown (wealth path, including starting-capital peak)

```python
def drawdown(ret):
    wealth = (1 + ret.dropna()).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1

def max_drawdown(ret):
    return float(drawdown(ret).min())
```
(`src/evaluate.py:83-93`.) Wealth is a $1 investment compounded through the return path,
$W_t = \prod_{s\le t}(1+r_s)$; the running peak is $\max(\text{cummax}(W_t), 1)$ — the `.clip(
lower=1.0)` is the important detail. Without it, `cummax()` alone would treat $W_1$ (after the
very first month's return) as the initial peak, so a loss in month 1 would show a drawdown of
exactly 0 (the series is "at its own peak" trivially). By clipping the peak floor to $1.0$ — the
starting capital itself — a first-month loss is immediately underwater relative to where the
strategy started. `max_drawdown` is then just $\min_t(\text{drawdown}_t)$, a negative number (or
0). `tests/test_evaluate.py::test_first_month_drawdown_includes_starting_capital` plants
`ret = [-0.05, 0.01]` and checks `drawdown.iloc[0] == -0.05` and `max_drawdown == -0.05` — proving
the starting-capital peak, not the post-loss wealth level, anchors the very first observation.

### 2.10 Correlation with S&P 500

`performance_table()` computes, for every row (`strategy_gross`, `strategy_net`, `benchmark`,
`sp500`): `returns[col].corr(returns['sp500_ret'])` (`src/evaluate.py:158-160`) — an ordinary
Pearson correlation of that row's own monthly return series against `sp500_ret`. The `sp500` row
therefore reports a correlation of exactly 1.0 with itself, which is intentional (a sanity check
a reader can eyeball) rather than a special case in the code.

### 2.11 Long/short leg contributions

```python
for leg, col in [('long_leg_excess_contrib', 'long_ret'), ('short_leg_excess_contrib', 'short_ret')]:
    df.loc[leg, ['avg_monthly', 'ann_arith', 'cumulative']] = [
        returns[col].mean(), returns[col].mean() * 12, cumulative(returns[col])]
```
(`src/evaluate.py:153-155`.) Two extra rows appended to `performance_table()`, reporting the
average monthly, arithmetic-annualized, and compounded-cumulative return of `long_ret` and
`short_ret` separately (both defined in `src/portfolio.py` as $\sum_{w>0}w\,r$ and
$\sum_{w<0}w\,r$ on the $\pm\$100$-per-leg convention), so a reader can see which leg is actually
carrying the strategy's performance, as the brief asks (page 21: "Return of the long leg and the
short leg reported separately, so the committee can see which side is actually working").

### 2.12 Gross vs. net-of-cost

`performance_table()` builds `strategy_gross` from `total_ret`/`active_ret` and `strategy_net`
from `total_ret_net`/`active_ret_net` (both columns computed upstream in `src/portfolio.py`:
`total_ret_net = total_ret - cost`, `cost = COST_BPS/1e4 \times \sum|w_t - w_{t-1}|`), and tags a
`cost_basis` column (`'gross'` everywhere except the `strategy_net` row, `src/evaluate.py:156-157`)
so every consumer of the table can tell which rows are which without guessing from the row name
alone. `run_evaluation()` additionally reports `ir_net = perf.loc['strategy_net', 'ir']` in the
headline dict (§7) alongside the gross `ir` — per the module docstring's own convention, "all
tables/charts are gross of trading costs unless marked net (a `cost_basis` column, a `_net`/
`_gross` suffix, or the chart title)" (`src/evaluate.py:26-27`).

---

## 3. Alpha and beta: the market-model regression

```python
def alpha_beta(returns, ret_col='total_ret'):
    y = (returns[ret_col] - returns['rf_m']).astype(float)
    x = sm.add_constant(returns['sp500_exret'].astype(float))
    ols = sm.OLS(y, x, missing='drop').fit()
    nw = sm.OLS(y, x, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3}, use_t=True)
    return {'alpha_m': ols.params['const'], 'alpha_ann': ols.params['const'] * 12,
            'beta': ols.params['sp500_exret'],
            'alpha_se_ols': ols.bse['const'], 'alpha_t_ols': ols.tvalues['const'],
            'beta_se_ols': ols.bse['sp500_exret'], 'beta_t_ols': ols.tvalues['sp500_exret'],
            'alpha_se_nw': nw.bse['const'], 'alpha_t_nw': nw.tvalues['const'],
            'beta_se_nw': nw.bse['sp500_exret'], 'beta_t_nw': nw.tvalues['sp500_exret'],
            'n_obs': int(ols.nobs)}
```
(`src/evaluate.py:100-114`.) This is the exact regression the brief specifies (page 15):

$$R_{p,t} - r_{f,t} = \alpha + \beta\,(R_{SP,t} - r_{f,t}) + \epsilon_t$$

with $R_{p,t}$ = `total_ret` (or `total_ret_net` for the net-of-cost version — `run_evaluation()`
computes both, `ab_gross = alpha_beta(returns, 'total_ret')` and
`ab_net = alpha_beta(returns, 'total_ret_net')`, `src/evaluate.py:434-435`), $r_{f,t}$ = `rf_m`,
and $R_{SP,t}-r_{f,t}$ = the pre-computed `sp500_exret` column from `src/data.py`'s
`load_market()`. $\hat\alpha$ = the intercept, $\hat\beta$ = the slope.

### 3.1 OLS SEs vs. Newey-West HAC SEs (3 lags)

Two regressions are fit on the *same* $(y,x)$ data, differing only in the covariance estimator:

- **`ols`**: the textbook OLS covariance, $\widehat{\text{Var}}(\hat\theta) =
  \hat\sigma^2 (X^\top X)^{-1}$, valid only if the residuals $\epsilon_t$ are i.i.d.
  (homoskedastic, uncorrelated across $t$).
- **`nw`**: `cov_type='HAC', cov_kwds={'maxlags': 3}` — a **Newey-West** heteroskedasticity- and
  autocorrelation-consistent (HAC) sandwich estimator, allowing residual variance to change over
  time *and* residuals up to 3 months apart to be correlated with each other.

**Why HAC for monthly portfolio returns.** Plain OLS standard errors are only correct if
$\text{Cov}(\epsilon_t, \epsilon_{t-k}) = 0$ for all $k \ne 0$. That assumption is exactly the one
this pipeline's own construction violates: `src/portfolio.py`'s `smooth()` applies a per-permno
exponential moving average across formation months (`s_t = \text{EMA\_ALPHA}\cdot z_t +
(1-\text{EMA\_ALPHA})\cdot s_{t-1}`, `docs/SPEC.md` section 6) specifically *to reduce turnover*,
which means this month's target positions are a weighted blend of several previous months'
signals — inducing serial correlation in *positions*, and hence in realized *returns*, across
adjacent months. A regression that ignored this would understate the true sampling uncertainty
of $\hat\alpha$ and $\hat\beta$ and overstate their t-statistics — exactly the failure mode HAC
is built to correct. The brief itself asks for "the standard error on beta as well as on alpha"
(page 15) without mandating HAC by name, but reporting both the plain-OLS and the HAC figures
side by side (as `alpha_beta()` does) lets a reader see how much the naive SE understates the
truth.

**The Bartlett kernel.** Newey-West estimates the long-run covariance of the moment conditions as
a weighted sum of the sample autocovariances at lags $0$ through $L$ (`maxlags=3` here), using
Bartlett (linearly-decaying triangular) weights:

$$\hat{S} = \hat\Gamma_0 + \sum_{k=1}^{L} w_k \left(\hat\Gamma_k + \hat\Gamma_k^\top\right),
\qquad w_k = 1 - \frac{k}{L+1}$$

The linear down-weighting (rather than an equal weight on every lag up to $L$) is what
guarantees the resulting covariance matrix estimate stays positive semi-definite for any sample —
a property a naive truncated-lag average does not have. `statsmodels`'s `cov_type='HAC'` uses
this Bartlett kernel by default.

### 3.2 t-statistics and annualizing alpha

`alpha_t_ols`/`alpha_t_nw` and `beta_t_ols`/`beta_t_nw` are the usual $\hat\theta / \text{SE}
(\hat\theta)$ ratios `statsmodels` computes directly (`ols.tvalues`, `nw.tvalues`), using a
$t$-distribution (`use_t=True` on the NW fit forces $t$, not normal, critical values, matching
the OLS fit's default). Since $\alpha$ is estimated from **monthly** excess returns, it is
naturally a monthly number; `alpha_ann = alpha_m \times 12` (`src/evaluate.py:107`) simply scales
it up arithmetically to an annual rate, exactly as the brief instructs ("When you want to
annualize Alpha... you can multiply it by 12," page 15). Note this is the same simple
$\times 12$ scaling as `ann_arith` in §2.1 — not a compounded $(1+\alpha_m)^{12}-1$ — so it is
consistent with, but conceptually distinct from, `cagr()`.

### 3.3 What the committee reads from beta ≈ 0

Per the brief (page 15): "The slope is your evidence of neutrality — over the full
out-of-sample period it should sit close to zero, and we will read a beta far from zero as a
failure of the mandate rather than a stylistic choice." Concretely, a reader checks whether
$\hat\beta$ is statistically indistinguishable from 0 by comparing it against roughly
$\pm 2\times\text{SE}(\hat\beta)$ (using the **HAC** SE, `beta_se_nw`, as the more defensible one
given §3.1) — i.e. whether $|\hat\beta|/\text{SE}_{\text{NW}}(\hat\beta) = $
`beta_t_nw` is small. `run_evaluation()` surfaces `beta` and `beta_se_nw` directly in the
headline dict (§7) for exactly this check. `tests/test_evaluate.py::test_alpha_beta_recovers_planted_values`
verifies the machinery on a synthetic series with a planted $\beta=0.1$, $\alpha=0.002$/month, and
checks both that the recovered `beta`/`alpha_m` are close to the planted values and that
`alpha_t_nw`/`beta_se_nw` come back finite (not `NaN`/`inf`) — i.e. the HAC fit runs cleanly on a
realistic-length series.

---

## 4. The IR–Sharpe–hurdle relationship, and why full 200% gross matters

`Hackathon_Strategy_Recommendation.pdf` (the project's internal strategy plan, page 1) states:

$$\text{IR} \approx \text{Sharpe(spread)} - \frac{\sqrt{12}\times(0.04/12)}{k\,\sigma_{\text{spread}}}$$

where $k$ is gross exposure and $\sigma_{\text{spread}}$ is the volatility of the long-short
spread return. This is a direct algebraic consequence of the strategy's own construction, and is
worth deriving explicitly against the code's own quantities, because it explains a design
decision — not a backtest result.

**Derivation.** In `src/portfolio.py`'s capital convention (`docs/SPEC.md` section 6), collateral
earns the risk-free rate exactly, so `total_ret = rf_m + ls_ret` and, because `bench_ret = rf_m +
0.04/12` uses the *same* `rf_m` series, the risk-free leg cancels exactly in the active return:

$$\text{active}_t = \text{total\_ret}_t - \text{bench\_ret}_t = \text{ls\_ret}_t - \frac{0.04}{12}$$

Now suppose (hypothetically) the strategy only deployed a fraction $k/2$ of its permitted 200%
gross (i.e. scaled every position by $k/2$, so a full $k=2$ recovers the mandate's actual 200%
cap). The realized spread return scales linearly with position size, $\text{ls\_ret}_t^{(k)} = k
\cdot \text{ls\_ret}_t^{(1)}$, and so does its standard deviation, $\sigma^{(k)} = k\,\sigma^{(1)}$
— but the $0.04/12$ hurdle is a **fixed absolute drag**, independent of how much gross exposure is
actually used. So:

$$\text{IR}^{(k)} = \frac{\sqrt{12}\left(k\,\overline{\text{ls\_ret}}^{(1)} - 0.04/12\right)}
{k\,\sigma^{(1)}}
= \underbrace{\frac{\sqrt{12}\,\overline{\text{ls\_ret}}^{(1)}}{\sigma^{(1)}}}_{\text{Sharpe(spread), scale-invariant}}
- \frac{\sqrt{12}\times(0.04/12)}{k\,\sigma^{(1)}}$$

**The intuition.** The Sharpe ratio of the raw long-short spread is *scale-invariant* — doubling
every position doubles both the numerator and the denominator, leaving the ratio unchanged. The
$0.04$-per-year hurdle, however, does *not* scale with position size: it is levied against $1 of
capital regardless of how aggressively that capital is deployed. Running at a smaller $k$ (using
less of the allowed 200% gross) therefore does nothing for the Sharpe component but makes the
fixed hurdle's IR *penalty* proportionally larger (dividing by a smaller $k$). Running at the
mandate's full $k=2$ is the only lever that shrinks this penalty without giving anything up —
which is exactly why `docs/SPEC.md` section 6 fixes `GROSS = 2.0` as a hard constraint in
`optimize_month()` (`sum(long) = 1, sum(short) = -1`, always exactly 200% gross by construction,
never a free choice) and why the strategy plan states the implication directly: "The 4% hurdle
is a fixed drag, so run the full 200% gross." In this codebase $k$ is not tuned or reported as a
free parameter — it is pinned at the mandate's ceiling by the optimizer's own equality
constraints, and `exposure_table()`'s `gross_avg/min/max` (§5.1) is the diagnostic that confirms
the backtest actually held that ceiling every month rather than drifting below it.

---

## 5. Tables

### 5.1 Exposure table — `exposure_table(returns, holdings)`

(`src/evaluate.py:176-206`, `docs/SPEC.md` section 7 + amendment A10.) One row per test month is
aggregated into a single `pd.Series` of statistics:

| Field | Definition |
|---|---|
| `avg_n_long`, `avg_n_short` | mean of `returns['n_long']`/`n_short` (position counts per leg) |
| `gross_avg/min/max` | mean/min/max of `returns['gross']` ($=\sum \lvert w \rvert$) |
| `net_avg/min/max` | mean/min/max of `returns['net']` ($=\sum w$) |
| `filer_net_avg/min/max` | mean/min/max of `returns['filer_net']` — see below |
| `relax_share`, `relax_count_<label>` | see below |
| `avg_abs_weight`, `max_abs_weight` | mean/max of `\lvert holdings['weight'] \rvert` across all held names, all months |
| `top10_share_gross_avg` | mean, over months, of (sum of the 10 largest `\lvert weight \rvert` that month) / (total `\lvert weight \rvert` that month) |
| `turnover_avg/min/max` | mean/min/max of `_turnover(returns)`, i.e. `returns['turnover']` with the first month excluded (§5.1.1) |
| `beta_exante_avg` | mean of `returns['beta_exante']` (the ex-ante portfolio beta, $\text{beta}\cdot w$, enforced $\le$ `BETA_TOL` by the optimizer) |
| `missing_ret_weight_avg` | mean of `returns['missing_ret_weight']` — the A4 diagnostic for how much absolute weight each month sat in names with a missing next-month return (zero-filled per A4) |

**`relax_share` / `filer_net` (A10 diagnostics).** `src/portfolio.py`'s `optimize_month()` first
tries to solve at the mandate's base tolerances (`SECTOR_TOL`, `SIZE_TOL`, `BETA_TOL`); if
infeasible, it relaxes a ladder of tolerances (`RELAX_STEPS`, `src/portfolio.py:11-16`: sector
$\times2$, then $+$size $\times2$, then $+$beta $\times2$) and records which relaxation, if any,
was needed as a string in `returns['relax']` (empty string `''` if none). `exposure_table()`
computes:

```python
relaxed = returns['relax'].astype(str) != ''
relax_stats = {'relax_share': relaxed.mean()}
relax_stats.update({f'relax_count_{label}': int(n) for label, n in relax_counts.items()})
```
(`src/evaluate.py:189-192`.) `relax_share` is simply the fraction of test months that needed *any*
relaxation; `relax_count_<label>` breaks that down by which specific ladder step string appeared.
This exists because `docs/research_log.md`'s 2026-09-27 entry documents that real data frequently
sits with the worst GICS2 sector's exposure exactly at the `SECTOR_TOL` boundary (a lopsided
candidate-set composition, not a solver bug), so relaxation is expected to happen non-trivially
often — `relax_share` quantifies exactly how often.

`filer_net` is a separate A10 diagnostic: `rec['filer_net'] = (w * has_filing).sum()`
(`src/portfolio.py:257`) — the net exposure to 8-K filers specifically. A10 adds a constraint
that $\lvert\sum_{\text{filers}} w\rvert \le \text{SECTOR\_TOL}$, because the text specialist's
within-filer-only z-score (non-filers get exactly 0, per A1) mechanically tilts the raw signal's
tails toward filers; `filer_net_avg/min/max` is how a reader checks that constraint actually held.

`tests/test_evaluate.py::test_exposure_table_relax_frequency_and_filer_net` and
`test_turnover_average_excludes_first_month` hand-verify `relax_share`, `relax_count_*`,
`filer_net_avg/min/max`, and the turnover figures against synthetic 4-month fixtures.

#### 5.1.1 Turnover excludes the first month

```python
def _turnover(returns):
    return returns.loc[~returns['first_month'], 'turnover']
```
(`src/evaluate.py:123-125`.) The very first backtest month starts from an *empty* prior book
(`w_prev` is empty), so its turnover figure — while a real, well-defined number, not `NaN` — is
not a like-for-like rebalance and would artificially inflate every turnover statistic if
included. `evaluate.py` filters it out via the `first_month` boolean flag (set upstream in
`src/portfolio.py`'s `backtest()`) rather than via `dropna()`, since the value itself is not
missing. Every function that reports turnover (`exposure_table`, `ablation_table`) routes
through this same helper.

### 5.2 Short-book table — `short_book_table(holdings, panel)`

(`src/evaluate.py:209-230`.) Restricted to `shorts = holdings[holdings['weight'] < 0]`, merged
onto the panel's `me`, `dolvol_126d_raw` (the **raw**, unranked dollar-volume column — see the
warning in the module docstring, `src/evaluate.py:10-11`: the panel's own `dolvol_126d` is
within-eom *ranked* to $[-1,1]$ and unusable for a dollar-level check), and `size_grp`:

| Field | Definition |
|---|---|
| `avg_me_weighted` | $\lvert w\rvert$-weighted average market equity of short names, `wavg('me')` |
| `avg_me_simple` | plain (unweighted) mean of `me` over short rows |
| `avg_dolvol_weighted` / `avg_dolvol_simple` | same, on `dolvol_126d_raw` |
| `share_nano_micro_weight` | fraction of *total short absolute weight* held in `size_grp` $\in$ `{'nano','micro'}` |
| `share_bottom_quintile_dolvol_weight` | fraction of total short absolute weight in names whose `dolvol_126d_raw` is $\le$ that month's 20th percentile **among the whole panel/universe** that month (`m['dolvol_q20'] = m['eom'].map(panel.groupby('eom')['dolvol_126d_raw'].quantile(0.20))`) |

`wavg(col)` (`src/evaluate.py:217-220`) divides by the total weight of rows with a *non-null*
value of that column only — so a stock missing `me` for some reason doesn't silently zero out the
weighted average, it is simply excluded from that particular average's denominator. The two
`share_*` fields divide by the *total* short book's weight regardless of nulls (they are asking
"what share of the whole short book"), which is a different denominator by design.

**Why these proxy hard-to-borrow names.** The brief flags this directly (page 8, truncated in
the extracted text but present): "shorting is not free. Borrow costs are real, hard-to-borrow
names can be recalled..." Small market-cap (`nano`/`micro` `size_grp`) and low dollar-trading-
volume stocks are the two most common, publicly-observable proxies for stocks that are
expensive or difficult to locate/borrow in practice — thin float, low institutional ownership,
and low liquidity all correlate with high borrow fees and recall risk, none of which this
backtest's frictionless short-selling assumption prices in. Reporting the short book's size and
liquidity profile lets a reader independently gauge how much of the strategy's short-side
performance might be optimistic relative to a real trading desk. This matches the brief's ask
directly (page 21: "Characteristics of the short book: average market capitalisation, average
dollar volume, and the share of short positions in small-cap or plausibly hard-to-borrow
names.").

`tests/test_evaluate.py::test_short_book_table_uses_raw_dolvol_not_ranked` specifically guards
against the ranked-column bug: it builds a synthetic panel where the *ranked* `dolvol_126d` would
sit in $[-1,1]$ but `dolvol_126d_raw` carries real dollar figures, and asserts
`avg_dolvol_weighted > 1000` — a value only reachable if the raw column, not the ranked one, was
actually used.

### 5.3 Top holdings — `top_holdings(holdings, n=10)`

(`src/evaluate.py:246-256`.) **Full-period average, including zeros**:

```python
n_months = holdings['month'].nunique()
avg_w = (holdings.groupby('permno')['weight'].sum() / n_months).rename('avg_weight')
```
A stock held in only 2 of the 68 test months at 4% weight gets `avg_weight = (0.04+0.04)/68`,
*not* `(0.04+0.04)/2` — dividing by the full month count, not by the number of months the name
was actually held. This is what "average weight over the test period" means here: a name has to
be *persistently* held to rank highly, not just briefly at a large size.
`tests/test_evaluate.py::test_top_holdings_averages_over_all_months_incl_zeros` verifies this
directly with a 4-month fixture where one permno is held in only 2 of 4 months.

Top-10 long and top-10 short are `tbl.sort_values('avg_weight', ascending=False).head(n)` and
`tbl.sort_values('avg_weight').head(n)` respectively — i.e. the most positive and most negative
average weights, which for a book that is long/short by construction naturally separates into
the "top 10 longs" and "top 10 shorts" the brief asks for (page 20, page 2 of the strategy plan).

### 5.4 Contributors — `contributors(holdings, panel, n=10)`

(`src/evaluate.py:233-243`.) Not average weight, but **total excess-return contribution**:

$$\text{contrib}_{\text{permno}} = \sum_{\text{months held}} w_{\text{permno},t}\cdot
\text{stock\_exret}_{\text{permno},t}$$

```python
m['contrib'] = m['weight'] * m['stock_exret'].fillna(0.0)
total = m.groupby('permno')['contrib'].sum().rename('total_contrib_gross')
```
`stock_exret` is filled with 0 for any missing value (consistent with the headline's A4
zero-fill convention for missing next-month returns), and this is always the **gross** (pre-cost)
contribution — there is no net-of-cost contributor table. `tests/test_evaluate.py::
test_contributors_sum_over_months` hand-verifies the summation on a 2-permno, 2-month fixture.

### 5.4.1 Label rule (shared by contributors and top_holdings)

```python
labels = holdings.sort_values('eom').groupby('permno')[['ticker', 'company_name']].last()
```
Both `contributors()` and `top_holdings()` label each `permno` using the ticker/company name from
the **latest `eom` at which that permno was actually held** — never a possibly-later label from
the panel that the stock was never actually held under, since `holdings` by construction only
contains rows where the stock was in the book. Sorting by `eom` then taking `.last()` per permno
picks the most recent held-period label, which is the more defensible convention for a "how was
this position labeled while we held it" audit trail (relevant to A8: deck-visible holdings whose
`label_source` is not the same-month panel label get independently verified against SEC EDGAR).

### 5.5 Regime table — `regime_table(returns)`

(`src/evaluate.py:259-270`, `REGIMES` dict at `src/evaluate.py:38-43`.) Four fixed periods:

```python
REGIMES = {'2021': ('2021-01-01', '2021-12-31'), '2022': ('2022-01-01', '2022-12-31'),
           '2023-2025': ('2023-01-01', '2025-12-31'), '2026': ('2026-01-01', '2026-12-31')}
```
For each, `sub = returns.loc[a:b]` (a label-based DatetimeIndex slice — automatically clipped to
whatever months exist, so the `'2026'` regime naturally only contains Jan–Aug 2026 even though
the range nominally runs through Dec 31), and:

| Field | Definition |
|---|---|
| `ir_gross` | `ir(sub['active_ret'])`, §2.7, computed within just that regime's months |
| `ann_active_gross` | `sub['active_ret'].mean() * 12` |
| `long_contrib_gross` | `sub['long_ret'].mean() * 12` |
| `short_contrib_gross` | `sub['short_ret'].mean() * 12` |
| `n_months` | `len(sub)` |

These four regimes map directly onto the strategy plan's characterization: "2021 rebound, 2022
drawdown, 2023–2025 concentration, 2026" — the point of this table is to let a reader see which
regime, and which leg (long vs. short), actually drove performance in each period, per the
brief's ask (page 21 implicitly, and the strategy plan page 4 explicitly: "Show which specialist
and which leg worked in each"). `tests/test_evaluate.py::test_regime_table_hand_computed` checks
`n_months`, `ann_active_gross`, `long_contrib_gross`, `short_contrib_gross` for the `'2021'` row
against a hand-constructed 14-month constant-return fixture.

### 5.6 Ablation table — `ablation_table(results)`

(`src/evaluate.py:273-284`.) `results` is a `dict {name: returns-shaped DataFrame}` — passed in
by `MAIN.py` as every alternative signal's own backtest output (`pred_gate`, `pred_gate_notext`,
`pred_ew_notext`, `pred_lgbm_all`, `pred_ridge`, and every `pred_spec_*` — see §8 below). Each
variant is independently run through `_restrict_window()` (its own 68-month completeness
assertion, §1.1) before any statistic is computed on it — so a broken ablation signal fails the
same way the headline would, rather than silently reporting over a partial window:

| Field | Definition |
|---|---|
| `ir` | `ir(r['active_ret'])` |
| `sharpe` | `sharpe(r['total_ret'], r['rf_m'])` |
| `ann_active` | `r['active_ret'].mean() * 12` |
| `beta` | `alpha_beta(r)['beta']` |
| `turnover` | `_turnover(r).mean()` |
| `max_dd` | `max_drawdown(r['total_ret'])` |

This directly answers the brief's ablation ask ("with and without text; gate versus equal
weighting; specialists versus a single model," strategy plan page 4) — `MAIN.py` step 6 supplies
exactly those comparisons as columns of `ablations`.

### 5.7 Sensitivity table

There is no dedicated `sensitivity_table()` function; `run_evaluation()` builds it inline
(`src/evaluate.py:451-458`) by running `sensitivity_returns` (from
`portfolio.missing_return_sensitivity`, an adverse -30%-longs/+30%-shorts stress on the A4
zero-filled missing-return names) through the *same* `performance_table()` +
`alpha_beta()` + `_headline_stats()` path as the real headline (§7), then writing
`pd.DataFrame([base_stats, sens_stats], index=['headline', 'sensitivity'])` to
`sensitivity_table.csv`. It is reported *alongside* the headline, never blended into it — the
module docstring is explicit about this (`src/evaluate.py:19-21`).

### 5.8 Gate weight series — `gate_weight_series(gate_coefs, state)`

(`src/evaluate.py:287-310`.) The effective weight the ridge gate assigns to specialist $k$ at
target month $t$ (formation month `eom`, target month $t=\text{eom}+1$, in test year $Y$ = the
year of $t$):

$$w_k(t) = b_k(Y) + \sum_{j} c_{kj}(Y)\cdot z_j(t; Y), \qquad
z_j(t; Y) = \frac{s_j(t) - \mu_j(Y)}{\sigma_j(Y)}$$

where $b_k(Y)$ is the `'base'`-term ridge coefficient for specialist $k$ in test year $Y$'s gate
fit, $c_{kj}(Y)$ is the coefficient on specialist $k$'s interaction with state variable $j$
(`config.STATE_VARS = ['mkt_vol12', 'disp']`), $s_j(t)$ is that state variable's raw value at
formation month `eom` (state is indexed by `eom`, mapped to `target_month = eom + 1` inside the
function), and $\mu_j(Y)$, $\sigma_j(Y)$ are the mean/std of $s_j$ over test year $Y$'s validation
window (`state_mean_<var>`/`state_std_<var>` columns already attached to `gate_coefs`, one value
per test year). In code:

```python
sd = s[f'state_std_{v}'].replace(0, np.nan)
s[f'z_{v}'] = ((s[v] - s[f'state_mean_{v}']) / sd).fillna(0.0)
...
merged['weight'] = merged['base'].fillna(0.0)
for v in state_vars:
    merged['weight'] += merged[f'z_{v}'] * merged[v].fillna(0.0)
```
A zero-std edge case (`sd.replace(0, np.nan)` then `.fillna(0.0)` on the resulting `NaN` z-score)
maps to a zero contribution from that state variable that month, rather than a division-by-zero
`inf`/`NaN` propagating into the weight. The whole computation is vectorized as two merges on
`test_year` (no per-month/per-specialist Python loop, per the function's own docstring), then
pivoted to `target_month × specialist` and restricted to the test window via `_window_mask`.
`tests/test_evaluate.py::test_gate_weight_series_hand_computed` hand-verifies the formula on a
single specialist/month: $z_{\text{mkt\_vol12}} = (0.20-0.15)/0.05 = 1.0$, $z_{\text{disp}} =
(0.03-0.02)/0.01 = 1.0$, weight $= 0.1 + 1.0\times0.05 + 1.0\times(-0.02) = 0.13$.

---

## 6. Every chart

All charts are saved via `_savefig()` (`src/evaluate.py:314-318`) to `config.FIG_DIR` at 150 dpi,
titled through `_title()` (`src/evaluate.py:321-322`) which appends `", gross of trading costs,
{PERIOD_LABEL}"` — `PERIOD_LABEL = "01/2021–08/2026"` (`src/evaluate.py:48`) — to every chart
title except the histogram's own explicit hurdle line and the gate-weight chart (which has no
gross/net distinction to make). Most charts prepend a synthetic zero-value point at
`_ANCHOR = 2020-12-31` (the month-end *before* the first test holding month) via `_anchored()`
(`src/evaluate.py:325-327`), so cumulative/underwater lines all visibly start from a common,
literal zero at the start of the backtest rather than jumping in already partway through a move.

| File | Function | What it shows | How to read it |
|---|---|---|---|
| `cumulative_returns.png` | `plot_cumulative` (`:344-347`) | Compounded cumulative return, $(1+r)$ cumprod $- 1$, anchored at 0 on 2020-12-31, for **Strategy** (`total_ret`), **Benchmark** (`bench_ret`), **S&P 500** (`sp500_ret`) | Three lines from a common zero; the vertical gap between Strategy and Benchmark at any date is (roughly) the compounded active return to that point; S&P 500 is shown for context only, per the brief, not as a performance target |
| `underwater.png` | `plot_underwater` (`:350-353`) | `drawdown()` (§2.9, includes the starting-capital peak) for **Strategy** and **S&P 500** | Always $\le 0$; the depth of each trough is how far below its own running peak that series was; comparing Strategy's troughs to the S&P 500's shows whether the market-neutral book actually avoided equity-market-style drawdowns |
| `rolling_active_return.png` | `plot_rolling_active` (`:356-359`) | 12-month rolling mean of `active_ret`, annualized ($\times 12$), hline at 0 | Smooths month-to-month noise; sustained time above the zero line is sustained outperformance of the cash+4% hurdle; below-zero stretches flag regimes (§5.5) where the strategy fell short |
| `rolling_ir.png` | `plot_rolling_ir` (`:362-366`) | 12-month rolling $\sqrt{12}\cdot\text{mean}(\text{active\_ret})/\text{std}(\text{active\_ret})$, hline at 0 | A rolling analog of the headline IR (§2.7) computed inline (not via `ir()`) on a trailing 12-month window; shows whether the strategy's risk-adjusted edge is stable or concentrated in a few windows |
| `rolling_beta.png` | `plot_rolling_beta` (`:369-373`) | 12-month rolling $\text{Cov}(R_p-r_f,\,R_{SP}-r_f)/\text{Var}(R_{SP}-r_f)$, hline at 0 | A rolling covariance/variance beta (not a rolling regression with intercept, and no HAC correction — it is a fast, windowed slope estimate only); per the brief, "the single clearest picture of whether you stayed neutral throughout, or only on average" (page 21) — a beta chart that averages to ~0 but swings widely within the window is a different (and worse) risk profile than one that stays near 0 throughout |
| `return_histogram.png` | `plot_return_histogram` (`:376-382`) | Histogram (20 bins) of monthly `total_ret`, with a dashed red vertical line at `bench_ret.mean()` labelled "Avg monthly hurdle" | The hurdle line is the **average** of the (slightly time-varying, TB3MS-linked) benchmark rate over the window, not a literal month-by-month hurdle; shows the distribution's shape (skew, fat tails) and what share of months cleared the (average) hurdle at a glance |
| `contributors.png` | `plot_contributors` (`:385-392`) | Horizontal bar chart of the top-10 and bottom-10 `total_contrib_gross` names from `contributors()` (§5.4), green if $\ge 0$ else red | Bars are gross excess-return contribution in dollars-per-$1-of-capital terms (i.e. the $w\cdot r$ sum), not weight — a name can be a top contributor with a modest average weight if it was held through a strong move |
| `gate_weights.png` | `plot_gate_weights` (`:395-398`) | `gate_weight_series()` (§5.8) output, one line per specialist, over the test window | Only produced when both `gate_coefs` and `state` are supplied to `run_evaluation()`; this is the "explainability exhibit" the strategy plan asks for (page 4: "A plot of the gate weights over time") — even though the gate itself is an *ablation*, not the headline signal, per A12 |

---

## 7. The headline dict, and the tables written to `outputs/tables`

### 7.1 `_headline_stats()`

```python
def _headline_stats(returns, perf, ab):
    """Built entirely from the performance table and alpha/beta results -- no recomputation."""
    return {'ir': perf.loc['strategy_gross', 'ir'], 'sharpe': perf.loc['strategy_gross', 'sharpe'],
            'cagr': perf.loc['strategy_gross', 'cagr'], 'max_dd': perf.loc['strategy_gross', 'max_dd'],
            'ir_net': perf.loc['strategy_net', 'ir'],
            'alpha_ann': ab['alpha_ann'], 'alpha_t_nw': ab['alpha_t_nw'],
            'beta': ab['beta'], 'beta_se_nw': ab['beta_se_nw'],
            'avg_turnover': _turnover(returns).mean(),
            'avg_n_long': returns['n_long'].mean(), 'avg_n_short': returns['n_short'].mean()}
```
(`src/evaluate.py:409-419`.) Note the docstring's own guarantee: every field here is *read off*
`performance_table()`'s `strategy_gross`/`strategy_net` rows or the `alpha_beta()` dict already
computed elsewhere in `run_evaluation()` — nothing is recomputed independently, so the headline
can never silently disagree with the tables it is summarizing.

### 7.2 `run_evaluation()`'s returned dict

```python
base_stats = _headline_stats(returns, perf, ab_gross)
headline = {**base_stats, 'ir_formula': IR_FORMULA}
if sensitivity_returns is not None:
    ...
    headline.update({f'sensitivity_{k}': v for k, v in sens_stats.items()})
```
(`src/evaluate.py:448-458`.) The final dict returned by `run_evaluation()` (and printed by
`MAIN.py`'s step 8, `src/evaluate.py:471` / `MAIN.py:144-145`) has 12 base keys (`ir`, `sharpe`,
`cagr`, `max_dd`, `ir_net`, `alpha_ann`, `alpha_t_nw`, `beta`, `beta_se_nw`, `avg_turnover`,
`avg_n_long`, `avg_n_short`), one string key `ir_formula`, and — whenever
`sensitivity_returns` is supplied (it always is, from `MAIN.py` step 5) — 12 more keys
`sensitivity_ir`, `sensitivity_sharpe`, ..., `sensitivity_avg_n_short`, mirroring the base 12
under the adverse missing-return stress (§5.7). `tests/test_evaluate.py::
test_run_evaluation_writes_tables_and_charts` enumerates and checks every one of these keys is
present and finite, plus `headline['ir_formula'] == ev.IR_FORMULA` and
`headline['sensitivity_ir'] < headline['ir']` (the synthetic fixture's stress case is constructed
to be strictly worse).

### 7.3 Tables written to `config.TABLE_DIR`

`run_evaluation()` writes the following CSVs (`src/evaluate.py:422-470`), all under
`outputs/tables/`:

| File | Written from |
|---|---|
| `performance_table.csv` | `performance_table(returns)` |
| `alpha_beta.csv` | `pd.DataFrame([ab_gross, ab_net], index=['gross', 'net'])` |
| `calendar_year_table.csv` | `calendar_year_table(returns)` |
| `exposure_table.csv` | `exposure_table(returns, holdings)` |
| `short_book_table.csv` | `short_book_table(holdings, panel)` |
| `contributors.csv` | `pd.concat([top_c, bot_c])` from `contributors()` |
| `top_holdings.csv` | `pd.concat([long_h, short_h], keys=['long', 'short'])` from `top_holdings()` |
| `regime_table.csv` | `regime_table(returns)` |
| `ablation_table.csv` | `ablation_table(ablations)` — only written if `ablations` is truthy |
| `sensitivity_table.csv` | the inline headline-vs-sensitivity frame (§5.7/§7.2) — only written if `sensitivity_returns` is not `None` |

Two related tables are written elsewhere, *not* by `evaluate.py`: `oos_r2.csv` (from
`models.r2_table()`, `MAIN.py` step 3) and `calibration.csv` (from `portfolio.calibrate()`,
`MAIN.py` step 4) — both land in `config.TABLE_DIR` too, but they are owned by `src/models.py`
and `src/portfolio.py` respectively, and are mentioned here only so a reader scanning
`outputs/tables/` isn't surprised to find files `evaluate.py` never wrote.

---

## 8. Tests, and what they prove

`tests/test_evaluate.py` (379 lines) is organized as one block per function/behavior, using small
synthetic fixtures rather than the real parquet data (per `docs/SPEC.md` section 0's testing
principle: "small synthetic data, run in seconds"). Grouped by what each block actually proves:

- **IR/Sharpe correctness and the zero-variance edge case** (`test_ir_hand_computed`,
  `test_ir_nan_when_zero_variance`, `test_sharpe_nan_when_zero_variance`,
  `test_sharpe_benchmark_zero_variance_is_nan_not_inf`): the $\sqrt{12}\,\overline{x}/s_x$ formula
  is correct against a hand computation, *and* a ratio with a zero (or ~zero) denominator returns
  `NaN`, never `inf`, regardless of whether the numerator is also zero or strictly positive — the
  latter is exactly the benchmark-row case (§2.8).
- **Alpha/beta regression recovery** (`test_alpha_beta_recovers_planted_values`): plants a known
  $\beta=0.1$, $\alpha=0.002$/month with Gaussian noise over 300 months and confirms both OLS
  point estimates land close to the planted values *and* the HAC (`alpha_t_nw`, `beta_se_nw`)
  outputs are finite — proving the Newey-West fit doesn't silently fail or diverge on realistic-
  length monthly data (§3).
- **CAGR / cumulative / max drawdown on known paths** (`test_cumulative_and_max_drawdown_known_path`,
  `test_cagr_constant_monthly_return`): closed-form checks against `[0.10, -0.50, 1.00]`
  and 12 months of a constant 1% return (§2.2–2.3, 2.9).
- **Calendar-year compounding and the 2026 YTD label** (`test_calendar_year_compounding`,
  `test_calendar_year_2026_labelled_ytd`): the per-year figure really is $\prod(1+r)-1$ within
  that year, and the 2026 row is labelled `"2026 YTD (Jan–Aug)"`, not presented as a full year
  (§2.6).
- **Contributors sum correctly and label correctly** (`test_contributors_sum_over_months`): the
  $\sum w\cdot r$ aggregation and the `"TICKER, Company Name"` label format are both exact on a
  2-name, 2-month fixture (§5.4).
- **Short book uses the raw, not ranked, dollar-volume column**
  (`test_short_book_table_uses_raw_dolvol_not_ranked`): guards specifically against a class of bug
  where the panel's within-eom-ranked `dolvol_126d` (range $[-1,1]$) gets used instead of
  `dolvol_126d_raw` — the weighted average would be nonsensically small if that regression
  happened (§5.2).
- **Turnover excludes the first month, via a flag not `dropna`**
  (`test_turnover_average_excludes_first_month`): the first month's turnover is a real,
  non-missing number that must still be excluded from every average/min/max because it reflects
  filling an empty book, not a rebalance (§5.1.1).
- **Exposure table's A10 diagnostics** (`test_exposure_table_relax_frequency_and_filer_net`):
  `relax_share`, `relax_count_<label>`, `filer_net_avg/min/max` computed correctly on a fixture
  where relaxation happens in exactly half the months (§5.1).
- **Drawdown includes the starting-capital peak** (`test_first_month_drawdown_includes_starting_capital`):
  a first-month $-5\%$ loss registers as a $-5\%$ drawdown immediately, not as 0
  (§2.9).
- **Benchmark row's ratios are all `NaN`** (`test_performance_table_benchmark_row_nan_ir_and_hit_rate`):
  `ir`, `hit_rate`, and `sharpe` are simultaneously `NaN` on the `benchmark` row of
  `performance_table()`, for the structural reasons in §2.7–2.8.
- **Top holdings average over all months, including zeros**
  (`test_top_holdings_averages_over_all_months_incl_zeros`): a name held in only 2 of 4 fixture
  months has its `avg_weight` computed over all 4 months, not just the 2 it was actually held in
  (§5.3).
- **Regime table hand-computed** (`test_regime_table_hand_computed`): `n_months`,
  `ann_active_gross`, `long_contrib_gross`, `short_contrib_gross` all check out on a 14-month
  constant-return fixture spanning the 2021/2022 boundary (§5.5).
- **Gate weight series hand-computed** (`test_gate_weight_series_hand_computed`): the
  $b_k + \sum_j c_{kj}\, z_j$ formula (§5.8) is verified to the fourth decimal on a single
  specialist/state-variable fixture.
- **Window-restriction assertion** (`test_restrict_window_asserts_on_missing_month`): deleting one
  month from a synthetic 68-month index makes `_restrict_window()` raise `AssertionError`,
  proving the completeness check (§1.1) actually fires rather than silently passing.
- **Full end-to-end orchestration** (`test_run_evaluation_writes_tables_and_charts`): a large
  synthetic bundle (`_synthetic_bundle()`, `:278-331` — 68 months, 10 permnos, 2 specialists,
  matching the real schema of `returns`/`holdings`/`panel`/`gate_coefs`/`state`) is run through
  the *actual* `run_evaluation()` with `monkeypatch` redirecting `config.FIG_DIR`/`TABLE_DIR` to a
  `tmp_path`, and the test asserts: every headline key is present and finite (§7.2), all 8 chart
  files exist, all 9 CSV tables exist (with `ablation_table.csv` correctly *absent* since no
  `ablations` dict is passed in this test), and specific new A4/A10 columns
  (`missing_ret_weight_avg`, `relax_share`, `filer_net_avg`) are present in the written
  `exposure_table.csv`. This is the test that proves the whole module runs end-to-end without a
  real backtest, on data shaped exactly like the real thing.

---

## Discrepancy noted between code and `docs/SPEC.md`

`docs/SPEC.md` section 7's prose description of `exposure_table()` (written before the section-10
amendments) lists only `avg n_long/n_short, gross avg/max, net avg/min/max, avg/max |weight|,
top-10 share of gross, turnover avg/min/max, ex-ante beta avg` — it omits `gross_min` (present in
code, consistent with A5's "reports ranges (min/avg/max)"), and entirely omits the A10 fields
(`filer_net_avg/min/max`, `relax_share`, `relax_count_<label>`) and the A4 field
(`missing_ret_weight_avg`), all four of which are implemented in `src/evaluate.py`, exercised by
`tests/test_evaluate.py`, and documented separately (correctly) in section 10's amendment text.
This is a stale/incomplete section-7 description, not a functional bug — the code and section 10
agree with each other; only section 7's prose hasn't been updated to match.
