"""Performance statistics, tables and charts for the test-window backtest (SPEC section 7).

Inputs, restricted to config.TEST_START..config.TEST_END (68 months, 2021-01..2026-08):
- `returns`: indexed by holding month `month` (src/portfolio.py), incl. total_ret_net,
  active_ret_net, missing_ret_weight and a bool `first_month` (empty prior book -> turnover
  there is a real number, not NaN, but excluded from every turnover figure via `_turnover()`).
  Always carries `relax` (str, '' if optimize_month needed no tolerance relaxation that month)
  and `filer_net` (A10 filer-net exposure diagnostic), read by exposure_table().
- `holdings`: month, eom, permno, weight, ticker, company_name, label_source.
- `panel`: permno, eom, stock_exret, raw `me`, `size_grp`, `dolvol_126d_raw` (the panel's own
  `dolvol_126d` is within-eom RANKED, unusable for short_book_table's dollar checks).
- `gate_coefs` (optional): one row per (test_year, specialist, term, coef), term 'base' (b_k)
  or a config.STATE_VARS name (c_kj), plus per-test_year `state_mean_<var>`/`state_std_<var>`.
  Effective weight w_k(t) = base_k(Y) + sum_j c_kj(Y) * standardized(s_j(t); Y), Y = test_year
  of t's target month.
- `state` (optional, paired with gate_coefs): indexed by eom, raw config.STATE_VARS columns.
- `ablations` (optional): dict {name: returns-shaped DataFrame}, each restricted/asserted the
  same way as `returns`.
- `sensitivity_returns` (optional, run_evaluation only): returns-shaped DataFrame from
  portfolio.missing_return_sensitivity, run through the same `_headline_stats` path and
  reported alongside (not blended into) the headline.

Zero-variance ratios: Sharpe/IR is NaN ("N/A") whenever the excess/active series has ~zero
std, regardless of its mean (e.g. the benchmark's own Sharpe, and its IR against itself).

All tables/charts are gross of trading costs unless marked net (a `cost_basis` column, a
`_net`/`_gross` suffix, or the chart title).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import config

REGIMES = {
    '2021': ('2021-01-01', '2021-12-31'),
    '2022': ('2022-01-01', '2022-12-31'),
    '2023-2025': ('2023-01-01', '2025-12-31'),
    '2026': ('2026-01-01', '2026-12-31'),
}

IR_FORMULA = ('IR = sqrt(12) * mean(active) / std(active), '
              'active = total_ret - (TB3MS/1200 + 0.04/12)')

PERIOD_LABEL = f"{config.TEST_START:%m/%Y}–{config.TEST_END:%m/%Y}"
_ANCHOR = config.TEST_START - pd.offsets.MonthEnd(1)  # 2020-12-31: pre-backtest chart anchor
_ZERO_STD_TOL = 1e-8  # below any realistic monthly-return std; floating noise on an exactly
# constant series (e.g. bench_ret - rf_m, or bench_ret vs itself) lands here.


# ---------------------------------------------------------------- low-level stats
def _ratio_or_nan(mean: float, sd: float) -> float:
    """sqrt(12)*mean/sd; undefined (NaN) whenever sd is ~0, never +/-inf."""
    if not np.isfinite(sd) or sd < _ZERO_STD_TOL:
        return np.nan
    return float(np.sqrt(12) * mean / sd)


def ir(active: pd.Series) -> float:
    active = active.dropna()
    return _ratio_or_nan(active.mean(), active.std(ddof=1))


def sharpe(ret: pd.Series, rf: pd.Series) -> float:
    excess = (ret - rf).dropna()
    return _ratio_or_nan(excess.mean(), excess.std(ddof=1))


def cagr(ret: pd.Series) -> float:
    # unreachable in this pipeline: every series reaching here is the (asserted) 68-month test
    # window with missing next-month returns already 0-filled (A4), never empty.
    ret = ret.dropna()
    return float((1 + ret).prod() ** (12 / len(ret)) - 1)


def cumulative(ret: pd.Series) -> float:
    return float((1 + ret.dropna()).prod() - 1)


def drawdown(ret: pd.Series) -> pd.Series:
    """Underwater series; starting capital (wealth=1) counts as a peak, so a month-1 loss is
    already a drawdown instead of being masked by cummax() starting at wealth[0]."""
    wealth = (1 + ret.dropna()).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1


def max_drawdown(ret: pd.Series) -> float:
    # unreachable in this pipeline, same as cagr() above: dd is never empty.
    return float(drawdown(ret).min())


def hit_rate(active: pd.Series) -> float:
    return float((active.dropna() > 0).mean())


def alpha_beta(returns: pd.DataFrame, ret_col: str = 'total_ret') -> dict:
    """OLS of (returns[ret_col] - rf_m) on sp500_exret; plain and Newey-West (3 lag) SEs."""
    y = (returns[ret_col] - returns['rf_m']).astype(float)
    x = sm.add_constant(returns['sp500_exret'].astype(float))
    ols = sm.OLS(y, x, missing='drop').fit()
    nw = sm.OLS(y, x, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3}, use_t=True)
    return {
        'alpha_m': ols.params['const'], 'alpha_ann': ols.params['const'] * 12,
        'beta': ols.params['sp500_exret'],
        'alpha_se_ols': ols.bse['const'], 'alpha_t_ols': ols.tvalues['const'],
        'beta_se_ols': ols.bse['sp500_exret'], 'beta_t_ols': ols.tvalues['sp500_exret'],
        'alpha_se_nw': nw.bse['const'], 'alpha_t_nw': nw.tvalues['const'],
        'beta_se_nw': nw.bse['sp500_exret'], 'beta_t_nw': nw.tvalues['sp500_exret'],
        'n_obs': int(ols.nobs),
    }


def _window_mask(dates) -> pd.Series:
    """Boolean mask for config.TEST_START..config.TEST_END: the one test-window filter, applied
    everywhere the test window is sliced (returns, holdings, gate weight series)."""
    return (dates >= config.TEST_START) & (dates <= config.TEST_END)


def _turnover(returns: pd.DataFrame) -> pd.Series:
    """Turnover excluding `first_month` (empty prior book, not a like-for-like rebalance)."""
    return returns.loc[~returns['first_month'], 'turnover']


def _labels(tbl: pd.DataFrame) -> pd.Series:
    return tbl['ticker'].astype(str) + ', ' + tbl['company_name'].astype(str)


# ---------------------------------------------------------------- tables
def _series_row(ret, active, rf):
    r = ret.dropna()
    return {
        'avg_monthly': ret.mean(), 'ann_arith': ret.mean() * 12, 'cagr': cagr(ret),
        'cumulative': cumulative(ret), 'sharpe': sharpe(ret, rf), 'max_dd': max_drawdown(ret),
        'ir': ir(active) if active is not None else np.nan,
        'hit_rate': hit_rate(active) if active is not None else np.nan,
        'best_month': r.idxmax(), 'best_val': float(r.max()),
        'worst_month': r.idxmin(), 'worst_val': float(r.min()),
    }


def performance_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = {
        'strategy_gross': _series_row(returns['total_ret'], returns['active_ret'], returns['rf_m']),
        'strategy_net': _series_row(returns['total_ret_net'], returns['active_ret_net'], returns['rf_m']),
        'benchmark': _series_row(returns['bench_ret'], None, returns['rf_m']),
        'sp500': _series_row(returns['sp500_ret'], returns['sp500_ret'] - returns['bench_ret'], returns['rf_m']),
    }
    df = pd.DataFrame(rows).T
    for leg, col in [('long_leg_excess_contrib', 'long_ret'), ('short_leg_excess_contrib', 'short_ret')]:
        df.loc[leg, ['avg_monthly', 'ann_arith', 'cumulative']] = [
            returns[col].mean(), returns[col].mean() * 12, cumulative(returns[col])]
    df['cost_basis'] = 'gross'
    df.loc['strategy_net', 'cost_basis'] = 'net'
    for label, col in [('strategy_gross', 'total_ret'), ('strategy_net', 'total_ret_net'),
                        ('benchmark', 'bench_ret'), ('sp500', 'sp500_ret')]:
        df.loc[label, 'corr_sp500'] = returns[col].corr(returns['sp500_ret'])
    return df


def calendar_year_table(returns: pd.DataFrame) -> pd.DataFrame:
    def compound(s):
        return (1 + s).prod() - 1
    g = returns.groupby(returns.index.year)
    out = pd.DataFrame({
        'strategy_gross': g['total_ret'].apply(compound), 'strategy_net': g['total_ret_net'].apply(compound),
        'benchmark': g['bench_ret'].apply(compound), 'sp500': g['sp500_ret'].apply(compound),
    })
    out.index = [f'{y} YTD (Jan–Aug)' if y == 2026 else str(y) for y in out.index]
    return out


def exposure_table(returns: pd.DataFrame, holdings: pd.DataFrame) -> pd.Series:
    turns = _turnover(returns)
    absw = holdings['weight'].abs()

    def top10_share(w):
        w = w.abs()
        tot = w.sum()
        return w.nlargest(10).sum() / tot if tot else np.nan

    top10 = holdings.groupby('month')['weight'].apply(top10_share)

    # A10 diagnostics: how often optimize_month had to relax a tolerance, and the filer-net
    # exposure the SECTOR_TOL-as-a-sector constraint is meant to keep small. The A15 cardinality
    # guard appends a 'cardinality' token to `relax` when it fires (src/portfolio.py); that's a
    # separate mechanism from the RELAX_STEPS tolerance ladder, so it's stripped out here and
    # reported on its own (cardinality_guard_share) rather than counted toward relax_share/
    # relax_count_*, which stay tolerance-relaxation-only.
    relax_raw = returns['relax'].astype(str)
    cardinality_guard = relax_raw.str.contains('cardinality')
    relax_tol = relax_raw.apply(
        lambda s: ','.join(tok for tok in s.split(',') if tok and tok != 'cardinality'))
    relaxed = relax_tol != ''
    relax_counts = relax_tol.loc[relaxed].value_counts()
    relax_stats = {'relax_share': relaxed.mean(), 'cardinality_guard_share': cardinality_guard.mean()}
    relax_stats.update({f'relax_count_{label}': int(n) for label, n in relax_counts.items()})

    return pd.Series({
        'avg_n_long': returns['n_long'].mean(), 'avg_n_short': returns['n_short'].mean(),
        'gross_avg': returns['gross'].mean(), 'gross_min': returns['gross'].min(), 'gross_max': returns['gross'].max(),
        'net_avg': returns['net'].mean(), 'net_min': returns['net'].min(), 'net_max': returns['net'].max(),
        'filer_net_avg': returns['filer_net'].mean(), 'filer_net_min': returns['filer_net'].min(),
        'filer_net_max': returns['filer_net'].max(),
        **relax_stats,
        'avg_abs_weight': absw.mean(), 'max_abs_weight': absw.max(),
        'top10_share_gross_avg': top10.mean(),
        'turnover_avg': turns.mean(), 'turnover_min': turns.min(), 'turnover_max': turns.max(),
        'beta_exante_avg': returns['beta_exante'].mean(),
        'missing_ret_weight_avg': returns['missing_ret_weight'].mean(),
    })


def short_book_table(holdings: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Weighted averages divide by the weight of rows with a non-null value only; the two
    share_* rows divide by total short weight (shares of the whole short book)."""
    shorts = holdings[holdings['weight'] < 0]
    m = shorts.merge(panel[['permno', 'eom', 'me', 'dolvol_126d_raw', 'size_grp']], on=['permno', 'eom'], how='left')
    w = m['weight'].abs()
    m['dolvol_q20'] = m['eom'].map(panel.groupby('eom')['dolvol_126d_raw'].quantile(0.20))

    def wavg(col):
        mask = m[col].notna()
        tw = w[mask].sum()
        return float((w[mask] * m.loc[mask, col]).sum() / tw) if tw else np.nan

    tot_w = w.sum()
    nano_micro = w[m['size_grp'].isin(['nano', 'micro'])].sum() / tot_w if tot_w else np.nan
    bottom_q = w[(m['dolvol_126d_raw'] <= m['dolvol_q20']).fillna(False)].sum() / tot_w if tot_w else np.nan
    return pd.Series({
        'avg_me_weighted': wavg('me'), 'avg_me_simple': m['me'].mean(),
        'avg_dolvol_weighted': wavg('dolvol_126d_raw'), 'avg_dolvol_simple': m['dolvol_126d_raw'].mean(),
        'share_nano_micro_weight': nano_micro,
        'share_bottom_quintile_dolvol_weight': bottom_q,
    })


def contributors(holdings: pd.DataFrame, panel: pd.DataFrame, n: int = 10):
    """Sum over months of weight * stock_exret (gross) by permno; top/bottom n with labels.
    Label is ticker/company_name from the latest `eom` the permno was actually HELD: `holdings`
    only contains held rows, so the groupby-last never picks a later out-of-holding label."""
    m = holdings.merge(panel[['permno', 'eom', 'stock_exret']], on=['permno', 'eom'], how='left')
    m['contrib'] = m['weight'] * m['stock_exret'].fillna(0.0)
    total = m.groupby('permno')['contrib'].sum().rename('total_contrib_gross')
    labels = holdings.sort_values('eom').groupby('permno')[['ticker', 'company_name']].last()
    tbl = labels.join(total).sort_values('total_contrib_gross', ascending=False)
    tbl['label'] = _labels(tbl)
    return tbl.head(n), tbl.tail(n).sort_values('total_contrib_gross')


def top_holdings(holdings: pd.DataFrame, n: int = 10):
    """Top n long / short names by average weight over ALL test months (sum of weight by
    permno / number of test months) -- zero for months a name isn't held. Label is
    ticker/company_name from the latest `eom` the permno was actually HELD (same as
    contributors()), never a later out-of-holding label."""
    n_months = holdings['month'].nunique()
    avg_w = (holdings.groupby('permno')['weight'].sum() / n_months).rename('avg_weight')
    labels = holdings.sort_values('eom').groupby('permno')[['ticker', 'company_name']].last()
    tbl = labels.join(avg_w)
    tbl['label'] = _labels(tbl)
    return tbl.sort_values('avg_weight', ascending=False).head(n), tbl.sort_values('avg_weight').head(n)


def regime_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, (a, b) in REGIMES.items():
        sub = returns.loc[a:b]
        if sub.empty:
            continue
        rows[name] = {
            'ir_gross': ir(sub['active_ret']), 'ann_active_gross': sub['active_ret'].mean() * 12,
            'long_contrib_gross': sub['long_ret'].mean() * 12, 'short_contrib_gross': sub['short_ret'].mean() * 12,
            'n_months': len(sub),
        }
    return pd.DataFrame(rows).T


def neutrality_table(returns: pd.DataFrame) -> pd.DataFrame:
    """The committee's first check: is this book actually market-neutral? One row per metric:
    full-period beta (OLS + Newey-West SE/t-stat, via alpha_beta), per-calendar-year beta (same,
    computed on each calendar year's rows only), average ex-ante beta, average/min/max net
    exposure, and correlation with the S&P. `se`/`t_stat` are populated for the beta rows only."""
    rows = {}
    full = alpha_beta(returns, 'total_ret')
    rows['beta_full_period'] = {'value': full['beta'], 'se': full['beta_se_nw'],
                                 't_stat': full['beta_t_nw'], 'n_obs': full['n_obs']}
    for year, sub in returns.groupby(returns.index.year):
        if len(sub) < 3:
            continue
        yb = alpha_beta(sub, 'total_ret')
        rows[f'beta_{year}'] = {'value': yb['beta'], 'se': yb['beta_se_nw'],
                                 't_stat': yb['beta_t_nw'], 'n_obs': yb['n_obs']}
    extra = {
        'avg_ex_ante_beta': returns['beta_exante'].mean(),
        'avg_net_exposure': returns['net'].mean(),
        'min_net_exposure': returns['net'].min(),
        'max_net_exposure': returns['net'].max(),
        'corr_sp500': returns['total_ret'].corr(returns['sp500_ret']),
    }
    for name, value in extra.items():
        rows[name] = {'value': value, 'se': np.nan, 't_stat': np.nan, 'n_obs': len(returns)}
    return pd.DataFrame(rows).T


def ablation_table(results: dict) -> pd.DataFrame:
    """Each variant is passed through the same test-window restriction/assert as `returns`."""
    rows = {}
    for name, r in results.items():
        r = _restrict_window(r)
        ab = alpha_beta(r)
        rows[name] = {
            'ir': ir(r['active_ret']), 'sharpe': sharpe(r['total_ret'], r['rf_m']),
            'ann_active': r['active_ret'].mean() * 12, 'beta': ab['beta'],
            'turnover': _turnover(r).mean(), 'max_dd': max_drawdown(r['total_ret']),
        }
    return pd.DataFrame(rows).T


def gate_weight_series(gate_coefs: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Effective weight b_k + sum_j c_kj * standardized(s_j(t)) per specialist k, by target
    month: one merge on test_year (vectorised, no per-month/per-specialist loop)."""
    state_vars = config.STATE_VARS
    stat_cols = [f'state_mean_{v}' for v in state_vars] + [f'state_std_{v}' for v in state_vars]
    stats = gate_coefs.groupby('test_year')[stat_cols].first().reset_index()
    wide = gate_coefs.pivot_table(index=['test_year', 'specialist'], columns='term', values='coef').reset_index()

    s = state.reset_index().rename(columns={'index': 'eom'})
    s['target_month'] = s['eom'] + pd.offsets.MonthEnd(1)
    s['test_year'] = s['target_month'].dt.year
    s = s.merge(stats, on='test_year', how='inner')
    for v in state_vars:
        sd = s[f'state_std_{v}'].replace(0, np.nan)
        s[f'z_{v}'] = ((s[v] - s[f'state_mean_{v}']) / sd).fillna(0.0)
    s = s[['target_month', 'test_year'] + [f'z_{v}' for v in state_vars]]

    merged = s.merge(wide, on='test_year', how='inner')
    merged['weight'] = merged['base'].fillna(0.0)
    for v in state_vars:
        merged['weight'] += merged[f'z_{v}'] * merged[v].fillna(0.0)

    out = merged.pivot_table(index='target_month', columns='specialist', values='weight')
    return out.loc[_window_mask(out.index)].sort_index()


# ---------------------------------------------------------------- charts
def _savefig(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=150)
    plt.close(fig)


def _title(base: str) -> str:
    return f"{base}, gross of trading costs, {PERIOD_LABEL}"


def _anchored(s: pd.Series) -> pd.Series:
    """Prepend a zero value at the pre-backtest anchor month (2020-12-31)."""
    return pd.concat([pd.Series([0.0], index=[_ANCHOR]), s])


def _line_plot(series_dict: dict, title: str, fname: str, hline=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, s in series_dict.items():
        ax.plot(s.index, s.values, label=label)
    if hline is not None:
        ax.axhline(hline, color='black', lw=0.8)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series_dict) > 1:
        ax.legend()
    _savefig(fig, fname)


def plot_cumulative(returns):
    series = {label: _anchored((1 + returns[col]).cumprod() - 1)
              for col, label in [('total_ret', 'Strategy'), ('bench_ret', 'Benchmark'), ('sp500_ret', 'S&P 500')]}
    _line_plot(series, _title('Cumulative return'), 'cumulative_returns.png', ylabel='Cumulative return')


def plot_underwater(returns):
    series = {'Strategy': _anchored(drawdown(returns['total_ret'])),
              'S&P 500': _anchored(drawdown(returns['sp500_ret']))}
    _line_plot(series, _title('Drawdown'), 'underwater.png', ylabel='Drawdown')


def plot_rolling_active(returns, window: int = 12):
    roll = returns['active_ret'].rolling(window).mean() * 12
    _line_plot({'Active return (ann.)': roll},
               _title(f'Rolling {window}-month annualized active return'), 'rolling_active_return.png', hline=0)


def plot_rolling_ir(returns, window: int = 12):
    m = returns['active_ret'].rolling(window).mean()
    sd = returns['active_ret'].rolling(window).std(ddof=1)
    _line_plot({'IR': np.sqrt(12) * m / sd},
               _title(f'Rolling {window}-month information ratio'), 'rolling_ir.png', hline=0)


def plot_rolling_beta(returns, window: int = 12):
    y = returns['total_ret'] - returns['rf_m']
    x = returns['sp500_exret']
    roll_beta = y.rolling(window).cov(x) / x.rolling(window).var()
    _line_plot({'Beta': roll_beta}, _title(f'Rolling {window}-month beta vs S&P 500'), 'rolling_beta.png', hline=0)


def plot_return_histogram(returns):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(returns['total_ret'], bins=20, color='tab:blue', alpha=0.85)
    ax.axvline(returns['bench_ret'].mean(), color='red', linestyle='--', label='Avg monthly hurdle')
    ax.set_title(_title('Distribution of monthly returns'))
    ax.legend()
    _savefig(fig, 'return_histogram.png')


def plot_contributors(top, bottom):
    both = pd.concat([top, bottom])
    colors = ['tab:green' if v >= 0 else 'tab:red' for v in both['total_contrib_gross']]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(both['label'], both['total_contrib_gross'], color=colors)
    ax.invert_yaxis()
    ax.set_title(_title('Top/bottom return contributors (excess-return contribution)'))
    _savefig(fig, 'contributors.png')


def plot_beta_by_year(neutrality: pd.DataFrame):
    """Bar chart of per-calendar-year beta vs S&P 500 (NW SE error bars), from neutrality_table's
    beta_YYYY rows -- the committee's first check, at a glance."""
    yearly = neutrality[neutrality.index.str.startswith('beta_') & (neutrality.index != 'beta_full_period')]
    years = [idx.replace('beta_', '') for idx in yearly.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(years, yearly['value'], yerr=yearly['se'], color='tab:blue', capsize=4)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Beta vs S&P 500')
    ax.set_title(f'Beta by calendar year, {PERIOD_LABEL}')
    _savefig(fig, 'beta_by_year.png')


def plot_gate_weights(gate_coefs, state):
    series = gate_weight_series(gate_coefs, state)
    _line_plot({c: series[c] for c in series.columns},
               f'Gate effective specialist weights, {PERIOD_LABEL}', 'gate_weights.png')


# ---------------------------------------------------------------- orchestration
def _restrict_window(returns: pd.DataFrame) -> pd.DataFrame:
    out = returns.loc[_window_mask(returns.index)].sort_index()
    missing = pd.date_range(config.TEST_START, config.TEST_END, freq='ME').difference(out.index)
    assert len(missing) == 0, f'evaluate: missing test-window months: {[d.date() for d in missing]}'
    return out


def _headline_stats(returns: pd.DataFrame, perf: pd.DataFrame, ab: dict) -> dict:
    """Built entirely from the performance table and alpha/beta results -- no recomputation."""
    return {
        'ir': perf.loc['strategy_gross', 'ir'], 'sharpe': perf.loc['strategy_gross', 'sharpe'],
        'cagr': perf.loc['strategy_gross', 'cagr'], 'max_dd': perf.loc['strategy_gross', 'max_dd'],
        'ir_net': perf.loc['strategy_net', 'ir'],
        'alpha_ann': ab['alpha_ann'], 'alpha_t_nw': ab['alpha_t_nw'],
        'beta': ab['beta'], 'beta_se_nw': ab['beta_se_nw'], 'beta_t_nw': ab['beta_t_nw'],
        'avg_turnover': _turnover(returns).mean(),
        'avg_n_long': returns['n_long'].mean(), 'avg_n_short': returns['n_short'].mean(),
    }


def run_evaluation(returns, holdings, panel, gate_coefs=None, state=None, ablations=None,
                    sensitivity_returns=None) -> dict:
    """sensitivity_returns: optional, same schema as `returns`, from
    portfolio.missing_return_sensitivity -- run through the same _headline_stats path as the
    headline, reported alongside it (not blended into it)."""
    returns = _restrict_window(returns)
    holdings = holdings[_window_mask(holdings['month'])]
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    print(IR_FORMULA)

    # neutrality check first -- this is the committee's first question, so it's the first table
    # computed, written and printed, ahead of performance/exposure/everything else.
    neutrality = neutrality_table(returns)
    neutrality.to_csv(config.TABLE_DIR / 'neutrality_table.csv')
    nf = neutrality.loc['beta_full_period']
    print('=== NEUTRALITY CHECK (committee first look) ===')
    print(f"full-period beta={nf['value']:.3f} (NW t-stat={nf['t_stat']:.2f}), "
          f"avg ex-ante beta={neutrality.loc['avg_ex_ante_beta', 'value']:.3f}, "
          f"avg net exposure={neutrality.loc['avg_net_exposure', 'value']:.3f} "
          f"(min={neutrality.loc['min_net_exposure', 'value']:.3f}, "
          f"max={neutrality.loc['max_net_exposure', 'value']:.3f}), "
          f"corr(S&P 500)={neutrality.loc['corr_sp500', 'value']:.3f}")

    perf = performance_table(returns)
    perf.to_csv(config.TABLE_DIR / 'performance_table.csv')
    ab_gross = alpha_beta(returns, 'total_ret')
    ab_net = alpha_beta(returns, 'total_ret_net')
    pd.DataFrame([ab_gross, ab_net], index=['gross', 'net']).to_csv(config.TABLE_DIR / 'alpha_beta.csv')
    calendar_year_table(returns).to_csv(config.TABLE_DIR / 'calendar_year_table.csv')
    exposure_table(returns, holdings).to_csv(config.TABLE_DIR / 'exposure_table.csv')
    short_book_table(holdings, panel).to_csv(config.TABLE_DIR / 'short_book_table.csv')
    top_c, bot_c = contributors(holdings, panel)
    pd.concat([top_c, bot_c]).to_csv(config.TABLE_DIR / 'contributors.csv')
    long_h, short_h = top_holdings(holdings)
    pd.concat([long_h, short_h], keys=['long', 'short']).to_csv(config.TABLE_DIR / 'top_holdings.csv')
    regime_table(returns).to_csv(config.TABLE_DIR / 'regime_table.csv')
    if ablations:
        ablation_table(ablations).to_csv(config.TABLE_DIR / 'ablation_table.csv')

    base_stats = _headline_stats(returns, perf, ab_gross)
    headline = {**base_stats, 'ir_formula': IR_FORMULA}

    if sensitivity_returns is not None:
        sensitivity_returns = _restrict_window(sensitivity_returns)
        sens_perf = performance_table(sensitivity_returns)
        sens_ab = alpha_beta(sensitivity_returns, 'total_ret')
        sens_stats = _headline_stats(sensitivity_returns, sens_perf, sens_ab)
        pd.DataFrame([base_stats, sens_stats], index=['headline', 'sensitivity']).to_csv(
            config.TABLE_DIR / 'sensitivity_table.csv')
        headline.update({f'sensitivity_{k}': v for k, v in sens_stats.items()})

    plot_cumulative(returns)
    plot_underwater(returns)
    plot_rolling_active(returns)
    plot_rolling_ir(returns)
    plot_rolling_beta(returns)
    plot_return_histogram(returns)
    plot_contributors(top_c, bot_c)
    plot_beta_by_year(neutrality)
    if gate_coefs is not None and state is not None:
        plot_gate_weights(gate_coefs, state)

    return headline
