import numpy as np
import pandas as pd
import pytest

from src import config
from src import evaluate as ev


# ---------------------------------------------------------------- IR / Sharpe
def test_ir_hand_computed():
    active = pd.Series([0.01, -0.02, 0.015, 0.005])
    expected = np.sqrt(12) * active.mean() / active.std(ddof=1)
    assert ev.ir(active) == pytest.approx(expected)


def test_ir_nan_when_zero_variance():
    # active is identically 0 (zero std, zero mean): still undefined, not "flat and zero".
    idx = pd.date_range('2021-01-31', periods=6, freq='ME')
    total = pd.Series(np.random.default_rng(0).normal(0.01, 0.02, 6), index=idx)
    bench = total.copy()
    active = total - bench
    assert np.isnan(ev.ir(active))


def test_sharpe_nan_when_zero_variance():
    idx = pd.date_range('2021-01-31', periods=6, freq='ME')
    rf = pd.Series(0.001, index=idx)
    ret = rf.copy()
    result = ev.sharpe(ret, rf)
    assert np.isnan(result)


def test_sharpe_benchmark_zero_variance_is_nan_not_inf():
    # bench_ret - rf_m is the constant hurdle 0.04/12: zero std, nonzero mean -> NaN, never inf.
    idx = pd.date_range('2021-01-31', periods=6, freq='ME')
    rf_m = pd.Series(0.001, index=idx)
    bench_ret = rf_m + config.HURDLE_ANNUAL / 12
    result = ev.sharpe(bench_ret, rf_m)
    assert np.isnan(result)
    assert not np.isinf(result)


# ---------------------------------------------------------------- alpha/beta
def test_alpha_beta_recovers_planted_values():
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.date_range('2000-01-31', periods=n, freq='ME')
    rf_m = pd.Series(0.001, index=idx)
    sp500_exret = pd.Series(rng.normal(0, 0.04, n), index=idx)
    noise = rng.normal(0, 0.01, n)
    total_ret = rf_m + 0.002 + 0.1 * sp500_exret + noise
    returns = pd.DataFrame({'total_ret': total_ret, 'rf_m': rf_m, 'sp500_exret': sp500_exret})
    result = ev.alpha_beta(returns, 'total_ret')
    assert result['beta'] == pytest.approx(0.1, abs=0.03)
    assert result['alpha_m'] == pytest.approx(0.002, abs=0.003)
    assert np.isfinite(result['alpha_t_nw'])
    assert np.isfinite(result['beta_se_nw'])


# ---------------------------------------------------------------- CAGR / cumulative / max DD
def test_cumulative_and_max_drawdown_known_path():
    ret = pd.Series([0.10, -0.50, 1.00])
    assert ev.cumulative(ret) == pytest.approx(0.10)
    assert ev.max_drawdown(ret) == pytest.approx(-0.50)


def test_cagr_constant_monthly_return():
    ret = pd.Series([0.01] * 12)
    expected = (1.01 ** 12) - 1
    assert ev.cagr(ret) == pytest.approx(expected)


# ---------------------------------------------------------------- calendar year
def test_calendar_year_compounding():
    idx = pd.date_range('2021-01-31', '2022-12-31', freq='ME')
    rng = np.random.default_rng(1)
    returns = pd.DataFrame({
        'total_ret': rng.normal(0.01, 0.02, len(idx)),
        'total_ret_net': rng.normal(0.008, 0.02, len(idx)),
        'bench_ret': rng.normal(0.005, 0.001, len(idx)),
        'sp500_ret': rng.normal(0.008, 0.03, len(idx)),
    }, index=idx)
    table = ev.calendar_year_table(returns)
    for year in (2021, 2022):
        sub = returns.loc[str(year)]
        expected = (1 + sub['total_ret']).prod() - 1
        assert table.loc[str(year), 'strategy_gross'] == pytest.approx(expected)


def test_calendar_year_2026_labelled_ytd():
    idx = pd.date_range('2025-01-31', '2026-08-31', freq='ME')
    rng = np.random.default_rng(2)
    returns = pd.DataFrame({
        'total_ret': rng.normal(0.01, 0.02, len(idx)),
        'total_ret_net': rng.normal(0.008, 0.02, len(idx)),
        'bench_ret': rng.normal(0.005, 0.001, len(idx)),
        'sp500_ret': rng.normal(0.008, 0.03, len(idx)),
    }, index=idx)
    table = ev.calendar_year_table(returns)
    assert '2026 YTD (Jan–Aug)' in table.index


# ---------------------------------------------------------------- contributors
def test_contributors_sum_over_months():
    holdings = pd.DataFrame({
        'month': pd.to_datetime(['2021-01-31', '2021-02-28', '2021-01-31', '2021-02-28']),
        'eom': pd.to_datetime(['2020-12-31', '2021-01-31', '2020-12-31', '2021-01-31']),
        'permno': [1, 1, 2, 2],
        'weight': [0.5, 0.3, -0.5, -0.3],
        'ticker': ['AAA', 'AAA', 'BBB', 'BBB'],
        'company_name': ['Alpha Co', 'Alpha Co', 'Beta Co', 'Beta Co'],
    })
    panel = pd.DataFrame({
        'permno': [1, 1, 2, 2],
        'eom': pd.to_datetime(['2020-12-31', '2021-01-31', '2020-12-31', '2021-01-31']),
        'stock_exret': [0.02, -0.01, 0.03, 0.04],
    })
    top, bottom = ev.contributors(holdings, panel, n=1)
    expected_1 = 0.5 * 0.02 + 0.3 * (-0.01)
    expected_2 = -0.5 * 0.03 + -0.3 * 0.04
    assert top.loc[1, 'total_contrib_gross'] == pytest.approx(expected_1)
    assert bottom.loc[2, 'total_contrib_gross'] == pytest.approx(expected_2)
    assert top.loc[1, 'label'] == 'AAA, Alpha Co'


# ---------------------------------------------------------------- short book uses raw dolvol
def test_short_book_table_uses_raw_dolvol_not_ranked():
    eom = pd.Timestamp('2021-01-31')
    holdings = pd.DataFrame({
        'month': [pd.Timestamp('2021-02-28')] * 3,
        'eom': [eom] * 3, 'permno': [1, 2, 3], 'weight': [-0.4, -0.3, -0.3],
    })
    # panel's ranked `dolvol_126d` would be in [-1, 1]; dolvol_126d_raw carries real dollar volume.
    panel = pd.DataFrame({
        'permno': [1, 2, 3, 4, 5], 'eom': [eom] * 5,
        'me': [1e9, 2e8, 5e7, 3e9, 4e9],
        'dolvol_126d_raw': [1e7, 1e5, 1e4, 1e8, 2e8],
        'size_grp': ['large', 'micro', 'nano', 'large', 'large'],
    })
    sb = ev.short_book_table(holdings, panel)
    # weight-averaged dolvol should reflect the raw dollar levels, not a [-1,1]-ranked scale.
    assert sb['avg_dolvol_weighted'] > 1000
    assert sb['share_nano_micro_weight'] == pytest.approx(0.6)


# ---------------------------------------------------------------- turnover excludes first month
def test_turnover_average_excludes_first_month():
    idx = pd.date_range('2021-01-31', periods=4, freq='ME')
    returns = pd.DataFrame({
        # first month's turnover is a real number (empty prior book), not NaN -- excluded via
        # the `first_month` flag, not via dropna().
        'turnover': [1.0, 0.2, 0.4, 0.6],
        'first_month': [True, False, False, False],
        'n_long': [10] * 4, 'n_short': [10] * 4,
        'gross': [1.8, 2.0, 1.9, 2.0], 'net': [0.0] * 4, 'beta_exante': [0.0] * 4,
        'missing_ret_weight': [0.0, 0.01, 0.02, 0.0],
        'relax': ['', '', '', ''], 'filer_net': [0.0] * 4,
    }, index=idx)
    holdings = pd.DataFrame({'month': idx.repeat(2), 'weight': [0.1, -0.1] * 4})
    exp = ev.exposure_table(returns, holdings)
    assert exp['turnover_avg'] == pytest.approx(np.mean([0.2, 0.4, 0.6]))
    assert exp['turnover_min'] == pytest.approx(0.2)
    assert exp['turnover_max'] == pytest.approx(0.6)
    assert exp['gross_min'] == pytest.approx(1.8)
    assert exp['gross_max'] == pytest.approx(2.0)
    assert exp['missing_ret_weight_avg'] == pytest.approx(np.mean([0.0, 0.01, 0.02, 0.0]))


# ---------------------------------------------------------------- exposure_table A10 diagnostics
def test_exposure_table_relax_frequency_and_filer_net():
    idx = pd.date_range('2021-01-31', periods=4, freq='ME')
    returns = pd.DataFrame({
        'turnover': [1.0, 0.2, 0.4, 0.6], 'first_month': [True, False, False, False],
        'n_long': [10] * 4, 'n_short': [10] * 4,
        'gross': [2.0] * 4, 'net': [0.0] * 4, 'beta_exante': [0.0] * 4,
        'missing_ret_weight': [0.0] * 4,
        'relax': ['', 'sector_tol_x2', 'sector_tol_x2', ''],
        'filer_net': [0.01, -0.02, 0.03, 0.0],
    }, index=idx)
    holdings = pd.DataFrame({'month': idx.repeat(2), 'weight': [0.1, -0.1] * 4})
    exp = ev.exposure_table(returns, holdings)
    assert exp['relax_share'] == pytest.approx(0.5)
    assert exp['relax_count_sector_tol_x2'] == 2
    assert exp['filer_net_avg'] == pytest.approx(np.mean([0.01, -0.02, 0.03, 0.0]))
    assert exp['filer_net_min'] == pytest.approx(-0.02)
    assert exp['filer_net_max'] == pytest.approx(0.03)


# ---------------------------------------------------------------- exposure_table A15 cardinality guard
def test_exposure_table_cardinality_stripped_from_relax_share():
    idx = pd.date_range('2021-01-31', periods=4, freq='ME')
    returns = pd.DataFrame({
        'turnover': [1.0, 0.2, 0.4, 0.6], 'first_month': [True, False, False, False],
        'n_long': [10] * 4, 'n_short': [10] * 4,
        'gross': [2.0] * 4, 'net': [0.0] * 4, 'beta_exante': [0.0] * 4,
        'missing_ret_weight': [0.0] * 4,
        # month 0: cardinality guard only fired, no tolerance relaxation.
        # month 1: tolerance relaxation AND the cardinality guard both fired.
        # month 2: tolerance relaxation only. month 3: neither.
        'relax': ['cardinality', 'sector_tol_x2,cardinality', 'sector_tol_x2', ''],
        'filer_net': [0.0] * 4,
    }, index=idx)
    holdings = pd.DataFrame({'month': idx.repeat(2), 'weight': [0.1, -0.1] * 4})
    exp = ev.exposure_table(returns, holdings)
    # relax_share/relax_count are tolerance-relaxation-only: only months 1 and 2 count, not month 0.
    assert exp['relax_share'] == pytest.approx(0.5)
    assert exp['relax_count_sector_tol_x2'] == 2
    # cardinality_guard_share counts months 0 and 1, regardless of tolerance relaxation.
    assert exp['cardinality_guard_share'] == pytest.approx(0.5)


# ---------------------------------------------------------------- drawdown incl. starting capital
def test_first_month_drawdown_includes_starting_capital():
    # Without treating starting capital ($1) as a peak, cummax() of wealth alone starts at
    # wealth[0] and a month-1 loss is invisible (dd[0] == 0 trivially).
    ret = pd.Series([-0.05, 0.01])
    assert ev.max_drawdown(ret) == pytest.approx(-0.05)
    dd = ev.drawdown(ret)
    assert dd.iloc[0] == pytest.approx(-0.05)


# ---------------------------------------------------------------- benchmark row: N/A ratios
def test_performance_table_benchmark_row_nan_ir_and_hit_rate():
    idx = pd.date_range('2021-01-31', periods=6, freq='ME')
    rf_m = pd.Series(0.001, index=idx)
    bench_ret = rf_m + config.HURDLE_ANNUAL / 12
    rng = np.random.default_rng(3)
    total_ret = rf_m + pd.Series(rng.normal(0.01, 0.02, 6), index=idx)
    total_ret_net = total_ret - 0.001
    active_ret = total_ret - bench_ret
    active_ret_net = total_ret_net - bench_ret
    sp500_ret = pd.Series(rng.normal(0.008, 0.03, 6), index=idx)
    returns = pd.DataFrame({
        'total_ret': total_ret, 'total_ret_net': total_ret_net, 'active_ret': active_ret,
        'active_ret_net': active_ret_net, 'bench_ret': bench_ret, 'rf_m': rf_m,
        'sp500_ret': sp500_ret, 'long_ret': total_ret / 2, 'short_ret': -total_ret / 2,
    }, index=idx)
    perf = ev.performance_table(returns)
    assert np.isnan(perf.loc['benchmark', 'ir'])
    assert np.isnan(perf.loc['benchmark', 'hit_rate'])
    assert np.isnan(perf.loc['benchmark', 'sharpe'])


# ---------------------------------------------------------------- top_holdings full-period average
def test_top_holdings_averages_over_all_months_incl_zeros():
    idx = pd.date_range('2021-01-31', periods=4, freq='ME')
    rows = [
        (idx[0], idx[0], 1, 0.04, 'AAA', 'Alpha Co'),
        (idx[1], idx[1], 1, 0.04, 'AAA', 'Alpha Co'),
        (idx[0], idx[0], 2, -0.01, 'BBB', 'Beta Co'),
        (idx[1], idx[1], 2, -0.01, 'BBB', 'Beta Co'),
        (idx[2], idx[2], 2, -0.01, 'BBB', 'Beta Co'),
        (idx[3], idx[3], 2, -0.01, 'BBB', 'Beta Co'),
    ]
    holdings = pd.DataFrame(rows, columns=['month', 'eom', 'permno', 'weight', 'ticker', 'company_name'])
    long_top, short_top = ev.top_holdings(holdings, n=2)
    # permno 1 held 0.04 in only 2 of 4 months -> average over ALL 4, not just the 2 held.
    assert long_top.loc[1, 'avg_weight'] == pytest.approx(0.08 / 4)
    assert short_top.loc[2, 'avg_weight'] == pytest.approx(-0.04 / 4)


# ---------------------------------------------------------------- regime_table hand-computed
def test_regime_table_hand_computed():
    idx = pd.date_range('2021-01-31', '2022-02-28', freq='ME')
    returns = pd.DataFrame({
        'active_ret': pd.Series(0.01, index=idx), 'long_ret': pd.Series(0.02, index=idx),
        'short_ret': pd.Series(-0.01, index=idx),
    }, index=idx)
    tbl = ev.regime_table(returns)
    assert tbl.loc['2021', 'n_months'] == 12
    assert tbl.loc['2021', 'ann_active_gross'] == pytest.approx(0.01 * 12)
    assert tbl.loc['2021', 'long_contrib_gross'] == pytest.approx(0.02 * 12)
    assert tbl.loc['2021', 'short_contrib_gross'] == pytest.approx(-0.01 * 12)


# ---------------------------------------------------------------- gate_weight_series hand-computed
def test_gate_weight_series_hand_computed():
    stats = {'state_mean_mkt_vol12': 0.15, 'state_std_mkt_vol12': 0.05,
             'state_mean_disp': 0.02, 'state_std_disp': 0.01}
    gate_coefs = pd.DataFrame([
        {'test_year': 2021, 'specialist': 'value', 'term': 'base', 'coef': 0.1, **stats},
        {'test_year': 2021, 'specialist': 'value', 'term': 'mkt_vol12', 'coef': 0.05, **stats},
        {'test_year': 2021, 'specialist': 'value', 'term': 'disp', 'coef': -0.02, **stats},
    ])
    state = pd.DataFrame({'mkt_vol12': [0.20], 'disp': [0.03]}, index=[pd.Timestamp('2020-12-31')])
    out = ev.gate_weight_series(gate_coefs, state)
    # z_mkt_vol12 = (0.20-0.15)/0.05 = 1.0, z_disp = (0.03-0.02)/0.01 = 1.0
    # weight = 0.1 + 1.0*0.05 + 1.0*(-0.02) = 0.13
    assert out.loc[pd.Timestamp('2021-01-31'), 'value'] == pytest.approx(0.13)


# ---------------------------------------------------------------- window restriction / assert
def test_restrict_window_asserts_on_missing_month():
    idx = pd.date_range(config.TEST_START, config.TEST_END, freq='ME').delete(5)
    returns = pd.DataFrame({'total_ret': np.zeros(len(idx))}, index=idx)
    with pytest.raises(AssertionError):
        ev._restrict_window(returns)


# ---------------------------------------------------------------- full synthetic fixture + charts
def _synthetic_bundle():
    idx = pd.date_range(config.TEST_START, config.TEST_END, freq='ME')
    n = len(idx)
    rng = np.random.default_rng(7)
    rf_m = pd.Series(0.0015, index=idx)
    sp500_ret = pd.Series(rng.normal(0.008, 0.04, n), index=idx)
    sp500_exret = sp500_ret - rf_m
    ls_ret = pd.Series(rng.normal(0.01, 0.03, n), index=idx)
    total_ret = rf_m + ls_ret
    bench_ret = rf_m + config.HURDLE_ANNUAL / 12
    active_ret = total_ret - bench_ret
    cost = pd.Series(np.r_[np.nan, rng.uniform(0.0005, 0.002, n - 1)], index=idx)
    total_ret_net = total_ret - cost.fillna(0.0)
    active_ret_net = total_ret_net - bench_ret
    returns = pd.DataFrame({
        'long_ret': ls_ret.clip(lower=0), 'short_ret': ls_ret.clip(upper=0) * 0 - 0.005,
        'ls_ret': ls_ret, 'total_ret': total_ret, 'bench_ret': bench_ret, 'active_ret': active_ret,
        'rf_m': rf_m, 'sp500_ret': sp500_ret, 'sp500_exret': sp500_exret,
        'n_long': 20, 'n_short': 20, 'gross': 2.0, 'net': 0.0, 'beta_exante': 0.0,
        'turnover': np.r_[1.0, rng.uniform(0.1, 0.5, n - 1)],
        'first_month': np.r_[True, np.full(n - 1, False)],
        'cost': cost, 'total_ret_net': total_ret_net, 'active_ret_net': active_ret_net,
        'missing_ret_weight': rng.uniform(0.0, 0.02, n),
        'relax': rng.choice(['', '', '', 'sector_tol_x2'], size=n),
        'filer_net': rng.uniform(-0.03, 0.03, n),
    }, index=idx)

    permnos = list(range(1, 11))
    rows = []
    for m in idx:
        w = rng.normal(0, 0.01, len(permnos))
        w = w - w.mean()
        for p, wi in zip(permnos, w):
            rows.append((m, m, p, wi, f'T{p}', f'Company {p}'))
    holdings = pd.DataFrame(rows, columns=['month', 'eom', 'permno', 'weight', 'ticker', 'company_name'])

    panel_rows = []
    for m in idx:
        for p in permnos:
            panel_rows.append((p, m, rng.normal(0.01, 0.05), 1e9, 5e6, rng.choice(['nano', 'micro', 'small', 'large'])))
    panel = pd.DataFrame(panel_rows, columns=['permno', 'eom', 'stock_exret', 'me', 'dolvol_126d_raw', 'size_grp'])

    specialists = ['value', 'momentum']
    gc_rows = []
    for year in sorted(idx.year.unique()):
        stats = {'state_mean_mkt_vol12': 0.15, 'state_std_mkt_vol12': 0.05,
                 'state_mean_disp': 0.02, 'state_std_disp': 0.01}
        for spec in specialists:
            for term, coef in [('base', 0.1), ('mkt_vol12', 0.05), ('disp', -0.02)]:
                gc_rows.append({'test_year': year, 'specialist': spec, 'term': term, 'coef': coef, **stats})
    gate_coefs = pd.DataFrame(gc_rows)
    state = pd.DataFrame({'mkt_vol12': rng.uniform(0.1, 0.2, n), 'disp': rng.uniform(0.01, 0.03, n)}, index=idx)

    return returns, holdings, panel, gate_coefs, state


def test_run_evaluation_writes_tables_and_charts(tmp_path, monkeypatch):
    fig_dir = tmp_path / 'figures'
    table_dir = tmp_path / 'tables'
    monkeypatch.setattr(config, 'FIG_DIR', fig_dir)
    monkeypatch.setattr(config, 'TABLE_DIR', table_dir)

    returns, holdings, panel, gate_coefs, state = _synthetic_bundle()
    # crude stand-in for portfolio.missing_return_sensitivity: same schema, worse active return.
    sensitivity_returns = returns.copy()
    sensitivity_returns['total_ret'] = returns['total_ret'] - 0.01
    sensitivity_returns['active_ret'] = sensitivity_returns['total_ret'] - returns['bench_ret']
    sensitivity_returns['active_ret_net'] = sensitivity_returns['active_ret'] - returns['cost'].fillna(0.0)
    sensitivity_returns['total_ret_net'] = sensitivity_returns['total_ret'] - returns['cost'].fillna(0.0)

    headline = ev.run_evaluation(returns, holdings, panel, gate_coefs=gate_coefs, state=state,
                                  sensitivity_returns=sensitivity_returns)

    for key in ('ir', 'ir_net', 'sharpe', 'alpha_ann', 'alpha_t_nw', 'beta', 'beta_se_nw',
                'max_dd', 'avg_turnover', 'cagr', 'avg_n_long', 'avg_n_short',
                'sensitivity_ir', 'sensitivity_sharpe', 'sensitivity_alpha_ann', 'sensitivity_beta'):
        assert key in headline
        assert np.isfinite(headline[key])
    assert headline['ir_formula'] == ev.IR_FORMULA
    assert headline['sensitivity_ir'] < headline['ir']

    expected_pngs = [
        'cumulative_returns.png', 'underwater.png', 'rolling_active_return.png',
        'rolling_ir.png', 'rolling_beta.png', 'return_histogram.png', 'contributors.png',
        'gate_weights.png',
    ]
    for name in expected_pngs:
        assert (fig_dir / name).exists()

    expected_csvs = [
        'performance_table.csv', 'alpha_beta.csv', 'calendar_year_table.csv',
        'exposure_table.csv', 'short_book_table.csv', 'contributors.csv',
        'top_holdings.csv', 'regime_table.csv', 'sensitivity_table.csv',
    ]
    for name in expected_csvs:
        assert (table_dir / name).exists()

    exposure = pd.read_csv(table_dir / 'exposure_table.csv', index_col=0)
    assert 'gross_min' in exposure.index
    assert 'missing_ret_weight_avg' in exposure.index
    assert 'relax_share' in exposure.index
    assert 'filer_net_avg' in exposure.index
