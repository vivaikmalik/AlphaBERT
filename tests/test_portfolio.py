import numpy as np
import pandas as pd
import pytest

from src import config, portfolio
from src.portfolio import (
    smooth, optimize_month, compute_month_return, attach_labels, write_submission,
    backtest, missing_return_sensitivity, _dust_and_rescale,
)

SECTORS = [f"{10 + 5 * i}" for i in range(11)]  # 11 synthetic GICS2 codes
N_STOCKS = 1500
N_MONTHS = 6


def make_month(rng, permnos):
    n = len(permnos)
    return pd.DataFrame({
        'permno': permnos,
        'signal': rng.normal(size=n),
        'beta': rng.normal(1.0, 0.3, size=n),
        'gics2': rng.choice(SECTORS, size=n),
        'size_z': rng.normal(size=n),
        'has_filing': rng.integers(0, 2, size=n),
    })


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def permnos():
    return np.arange(1, N_STOCKS + 1)


@pytest.fixture
def months(rng, permnos):
    return [make_month(rng, permnos) for _ in range(N_MONTHS)]


# ---------------------------------------------------------- optimize_month
def _check_month(w, relax, m):
    assert isinstance(relax, str)
    n_names = w.shape[0]
    assert 100 <= n_names <= 500

    long_w = w[w > 0]
    short_w = w[w < 0]
    assert long_w.sum() == pytest.approx(1.0, abs=1e-5)
    assert short_w.sum() == pytest.approx(-1.0, abs=1e-5)
    assert w.sum() == pytest.approx(0.0, abs=1e-5)
    assert w.abs().sum() == pytest.approx(2.0, abs=1e-5)
    assert w.abs().max() <= config.MAX_WEIGHT + 1e-5

    mi = m.set_index('permno')
    beta = mi['beta'].reindex(w.index).fillna(0.0)
    size_z = mi['size_z'].reindex(w.index).fillna(0.0)
    sector = mi['gics2'].reindex(w.index).fillna('NA')
    filer = mi['has_filing'].reindex(w.index).fillna(0)

    assert abs((w * beta).sum()) <= config.BETA_TOL + 1e-5
    assert abs((w * size_z).sum()) <= config.SIZE_TOL + 1e-5
    for s in sector.unique():
        assert abs(w[sector == s].sum()) <= config.SECTOR_TOL + 1e-5
    assert abs((w * filer).sum()) <= config.SECTOR_TOL + 1e-5


def test_optimize_month_constraints(months):
    w_prev = pd.Series(dtype=float)
    for m in months:
        w, relax = optimize_month(m, w_prev)
        _check_month(w, relax, m)
        w_prev = w


def test_dust_and_rescale_clips_post_rescale_overshoot():
    """A leg where one name sits at MAX_WEIGHT + 4e-7 after the raw rescale (the
    rescale ratio can nudge a near-cap name just over it, e.g. to 1.500004% > 1.5%)
    must come out with max |w| exactly <= MAX_WEIGHT, and the leg must still sum to
    exactly 1 (within 1e-12) after the clipped excess is redistributed."""
    cap = config.MAX_WEIGHT
    n = 120
    wv = np.full(n, 1.0 / n)
    wv[0] = cap + 4e-7
    # renormalize the rest so the raw (pre-fix) leg sum is exactly 1, mimicking what
    # optimize_month's solver output looks like just before _dust_and_rescale runs
    wv[1:] = (1.0 - wv[0]) / (n - 1)
    is_long = np.ones(n, dtype=bool)
    is_short = np.zeros(n, dtype=bool)

    out = _dust_and_rescale(wv, is_long, is_short)

    assert np.abs(out).max() <= cap + 1e-12
    assert out[is_long].sum() == pytest.approx(1.0, abs=1e-12)


def test_turnover_penalty_reduces_turnover(months):
    m1, m2 = months[0], months[1]
    w1, _ = optimize_month(m1, pd.Series(dtype=float))

    w2_low, _ = optimize_month(m2, w1, tc=0.0)
    w2_high, _ = optimize_month(m2, w1, tc=5.0)

    def turnover(w_new, w_old):
        idx = w_new.index.union(w_old.index)
        dw = w_new.reindex(idx).fillna(0.0) - w_old.reindex(idx).fillna(0.0)
        return 0.5 * dw.abs().sum() / config.GROSS

    assert turnover(w2_high, w1) < turnover(w2_low, w1)


def _correlated_filer_month(rng, permnos):
    """A month whose signal is correlated (not deterministic) with has_filing via a shared
    latent factor, leaving enough non-filers among the top candidates and filers among the
    bottom candidates that a filer-neutral book is feasible (verified empirically)."""
    n = len(permnos)
    latent = rng.normal(size=n)
    signal = latent + rng.normal(0, 1.8, size=n)
    m = pd.DataFrame({
        'permno': permnos, 'signal': signal,
        'beta': rng.normal(1.0, 0.3, size=n),
        'gics2': rng.choice(SECTORS, size=n),
        'size_z': rng.normal(size=n),
        'has_filing': (latent > 0).astype(int),
    })
    return m


def test_optimize_month_filer_net_neutral(rng, permnos):
    """Signal correlated with has_filing (A10, mandatory): |filer_net| stays within
    SECTOR_TOL even though the signal would otherwise push filers into the tails."""
    m_hf = _correlated_filer_month(rng, permnos)

    w, relax = optimize_month(m_hf, pd.Series(dtype=float))
    _check_month(w, relax, m_hf)  # existing constraints still hold, plus the filer check inside

    mi = m_hf.set_index('permno')
    filer_net = (w * mi['has_filing'].reindex(w.index).fillna(0)).sum()
    assert abs(filer_net) <= config.SECTOR_TOL + 1e-5


def test_optimize_month_forces_sector_relaxation():
    """Engineered month where the base tolerance is infeasible by construction but the x2
    relaxation step is feasible. Forces it through the has_filing==1 group rather than a
    gics2 sector: the filer group shares SECTOR_TOL and the same relaxation step as every
    gics2 sector (A10, `_tol_groups`), but -- unlike gics2 -- it is never touched by A14's
    within-gics2 signal demeaning, so all rows here share one gics2 value (demeaning is
    then a pure constant shift, no effect on ranking) and this construction stays valid
    regardless of the A14 fix. The filer group holds all but `m` of the long candidates and
    none of the short candidates, where `m` is picked (from config.N_CAND/MAX_WEIGHT) so the
    long leg's sum-to-1 constraint, combined with MAX_WEIGHT capping the `m` non-filer
    candidates, forces the filer group's net exposure to be strictly above SECTOR_TOL but at
    or below 2*SECTOR_TOL -- infeasible at the base tolerance for ANY choice of weights, not
    just for this signal. Regression test for the bug where _check_constraints asserted
    against the unrelaxed config tolerances even when solve_ladder only found a feasible
    solution at a relaxed step (real-data crash: 'sector x2' needed, assert fired against
    the base SECTOR_TOL anyway)."""
    cap = config.MAX_WEIGHT
    n_cand = config.N_CAND
    # m non-filer long candidates, capped at `cap` each, can cover at most m*cap of the long
    # leg's sum-to-1; the rest (1 - m*cap) is unavoidably the filer group. Pick m so that
    # forced exposure lands strictly between SECTOR_TOL and 2*SECTOR_TOL.
    target_min = 1.5 * config.SECTOR_TOL
    m = round((1 - target_min) / cap)
    n_filer_long = n_cand - m
    forced_min = 1 - m * cap
    assert config.SECTOR_TOL < forced_min <= 2 * config.SECTOR_TOL, \
        "test construction assumption violated for current config values"

    middle = 500
    n = 2 * n_cand + middle
    permno = np.arange(1, n + 1)
    signal = np.linspace(10, -10, n)  # strictly decreasing -> unambiguous top/bottom N_CAND

    has_filing = np.zeros(n, dtype=int)
    has_filing[:n_filer_long] = 1  # top of the long candidates: forced-infeasible filer group
    # everyone else (rest of long candidates, non-candidates, all short candidates) is 0

    m_df = pd.DataFrame({
        'permno': permno, 'signal': signal, 'beta': 1.0, 'gics2': '10', 'size_z': 0.0,
        'has_filing': has_filing,
    })

    w, relax = optimize_month(m_df, pd.Series(dtype=float))
    assert relax == 'sector x2'

    mi = m_df.set_index('permno')
    filer_w = mi['has_filing'].reindex(w.index).fillna(0)
    beta = mi['beta'].reindex(w.index).fillna(0.0)
    size_z = mi['size_z'].reindex(w.index).fillna(0.0)

    # the filer group could not have been kept within the base tolerance (that's the
    # point), but must sit within the relaxed one; beta/size were never relaxed
    # (RELAX_STEPS' 'sector x2' step only widens the sector/filer multiplier) so they stay
    # checked at their base tolerances.
    assert abs(w[filer_w == 1].sum()) > config.SECTOR_TOL + 1e-5
    assert abs(w[filer_w == 1].sum()) <= 2 * config.SECTOR_TOL + 1e-5
    assert abs(w[filer_w == 0].sum()) <= 2 * config.SECTOR_TOL + 1e-5
    assert abs((w * beta).sum()) <= config.BETA_TOL + 1e-5
    assert abs((w * size_z).sum()) <= config.SIZE_TOL + 1e-5

    long_w, short_w = w[w > 0], w[w < 0]
    assert long_w.sum() == pytest.approx(1.0, abs=1e-5)
    assert short_w.sum() == pytest.approx(-1.0, abs=1e-5)
    assert w.abs().max() <= cap + 1e-5
    assert 100 <= w.shape[0] <= 500


def test_optimize_month_sector_lopsided_demeaning_fixes_feasibility():
    """Regression for A14 (Opus audit): real formation months can have top/bottom N_CAND
    candidates so sector-lopsided by raw signal level that no relaxation step can satisfy
    sector neutrality (e.g. 2019-05-31: 147/250 shorts in GICS 35, 127/250 longs in GICS 40;
    minimum achievable max-sector net = 9.85% > the ladder's max of 2*SECTOR_TOL = 6%).
    optimize_month now selects candidates (and builds the objective) from the signal
    demeaned within gics2 -- for a sector whose raw signal level is high/low only because of
    an additive per-sector shift (not genuine within-sector dispersion), demeaning removes it
    from the candidate tails entirely. This synthetic month reproduces that shape: sector
    'S1' is shifted very negative (would dominate short candidates under the raw signal),
    sector 'S2' very positive (would dominate long candidates); many small filler sectors
    (baseline 0) carry the rest of the universe."""
    rng = np.random.default_rng(99)
    cap = config.MAX_WEIGHT
    n_cand = config.N_CAND

    n_shift = 200  # S1/S2 population; each below n_cand so it alone can't fill a whole leg
    n_filler_sectors = 20
    n_filler_each = 40
    n = 2 * n_shift + n_filler_sectors * n_filler_each

    permno = np.arange(1, n + 1)
    gics2 = np.empty(n, dtype=object)
    baseline = np.zeros(n)

    gics2[:n_shift] = 'S1'
    baseline[:n_shift] = -5.0  # raw signal: would dominate the short candidates
    gics2[n_shift:2 * n_shift] = 'S2'
    baseline[n_shift:2 * n_shift] = 5.0  # raw signal: would dominate the long candidates
    for k in range(n_filler_sectors):
        lo = 2 * n_shift + k * n_filler_each
        hi = lo + n_filler_each
        gics2[lo:hi] = f'F{k}'
        # baseline 0: filler sectors carry only noise, no systematic shift

    noise = rng.normal(scale=0.5, size=n)
    signal = baseline + noise

    # --- raw-signal candidate selection would be infeasible even after the sector x2 step:
    # at most `non_s1_in_short` short candidates can come from outside S1 (capped at `cap`
    # each); S1's forced net short exposure is then >= 1 - non_s1_in_short*cap. S1's baseline
    # (-5) is far below every other sector's (~0 or +5), so essentially none of the raw short
    # candidates are non-S1.
    order = np.argsort(signal)
    raw_short_cand_sectors = gics2[order[:n_cand]]
    non_s1_in_short = int((raw_short_cand_sectors != 'S1').sum())
    forced_min_s1 = 1 - non_s1_in_short * cap
    assert forced_min_s1 > 2 * config.SECTOR_TOL, \
        "test construction assumption violated: raw signal should force S1 past the ladder max"

    m_df = pd.DataFrame({
        'permno': permno, 'signal': signal, 'beta': 1.0, 'gics2': gics2, 'size_z': 0.0,
        'has_filing': 0,
    })

    # --- optimize_month demeans within gics2 before candidate selection and in the
    # objective (A14): S1/S2's additive shift is removed, leaving only noise-scale
    # within-sector deviations -- comparable to the filler sectors' -- so candidates (and
    # therefore sector exposures) come out balanced: feasible at base or one relax step.
    w, relax = optimize_month(m_df, pd.Series(dtype=float))
    assert relax in ('', 'sector x2')
    _check_month(w, relax, m_df)


def test_noncandidate_prev_holding_is_sold(months):
    m1, m2 = months[0], months[1]
    w1, _ = optimize_month(m1, pd.Series(dtype=float))
    held_permno = w1.index[0]

    # drop that name from month 2 entirely -> it cannot be a candidate
    m2_dropped = m2[m2['permno'] != held_permno]
    w2, _ = optimize_month(m2_dropped, w1)

    assert held_permno not in w2.index or w2[held_permno] == 0.0


# ---------------------------------------------------------- accounting
def test_compute_month_return_hand_computed():
    weights = pd.Series({1: 0.6, 2: 0.4, 3: -0.7, 4: -0.3})
    w_prev = pd.Series({1: 0.6, 2: 0.4, 3: -0.7, 4: -0.2, 5: -0.1})
    ret = pd.Series({1: 0.05, 2: -0.02, 3: 0.01, 4: np.nan})
    beta_map = pd.Series({1: 1.0, 2: 1.2, 3: 0.9, 4: 1.1})
    rf_m = 0.002
    sp500_ret = 0.03
    sp500_exret = sp500_ret - rf_m

    rec = compute_month_return(weights, w_prev, ret, rf_m, sp500_ret, sp500_exret, beta_map)

    assert rec['long_ret'] == pytest.approx(0.6 * 0.05 + 0.4 * -0.02)
    assert rec['short_ret'] == pytest.approx(-0.7 * 0.01 + -0.3 * 0.0)
    assert rec['ls_ret'] == pytest.approx(rec['long_ret'] + rec['short_ret'])
    assert rec['total_ret'] == pytest.approx(rf_m + rec['ls_ret'])
    assert rec['bench_ret'] == pytest.approx(rf_m + config.HURDLE_ANNUAL / 12)
    assert rec['active_ret'] == pytest.approx(rec['total_ret'] - rec['bench_ret'])
    assert rec['missing_ret_weight'] == pytest.approx(0.3)
    assert rec['n_long'] == 2 and rec['n_short'] == 2
    assert rec['gross'] == pytest.approx(2.0)
    assert rec['net'] == pytest.approx(0.0)

    dw_sum = 0.1 + 0.1  # |w4 change| + |w5 fully sold|
    expected_turnover = 0.5 * dw_sum / config.GROSS
    expected_cost = config.COST_BPS / 1e4 * dw_sum
    assert rec['turnover'] == pytest.approx(expected_turnover)
    assert rec['cost'] == pytest.approx(expected_cost)
    assert rec['total_ret_net'] == pytest.approx(rec['total_ret'] - expected_cost)
    assert rec['active_ret_net'] == pytest.approx(rec['active_ret'] - expected_cost)

    expected_beta = 0.6 * 1.0 + 0.4 * 1.2 + -0.7 * 0.9 + -0.3 * 1.1
    assert rec['beta_exante'] == pytest.approx(expected_beta)


def test_missing_return_sensitivity_hand_computed():
    """missing_return_sensitivity adjusts the headline `returns` (0-filled for missing
    stock_exret) by adding, per month, sum(weight*fill) over the missing names -- it must
    not re-run the accounting loop or touch turnover/cost."""
    month = pd.Timestamp('2021-02-28')
    holdings = pd.DataFrame({
        'month': [month] * 4,
        'eom': pd.to_datetime(['2021-01-31'] * 4),
        'permno': [1, 2, 3, 4],
        'weight': [0.6, 0.4, -0.7, -0.3],
    })
    panel = pd.DataFrame({
        'permno': [1, 2, 3, 4],
        'eom': pd.to_datetime(['2021-01-31'] * 4),
        'stock_exret': [0.05, np.nan, 0.01, np.nan],  # permno 2 (long) and 4 (short) missing
    })
    # headline returns as backtest() would have produced them: missing names 0-filled, so
    # long_ret/short_ret here reflect only the non-missing names.
    headline_long_ret = 0.6 * 0.05 + 0.4 * 0.0
    headline_short_ret = -0.7 * 0.01 + -0.3 * 0.0
    returns = pd.DataFrame({
        'long_ret': [headline_long_ret], 'short_ret': [headline_short_ret],
        'ls_ret': [headline_long_ret + headline_short_ret],
        'rf_m': [0.002], 'bench_ret': [0.005],
        'total_ret': [0.002 + headline_long_ret + headline_short_ret],
        'active_ret': [0.002 + headline_long_ret + headline_short_ret - 0.005],
        'cost': [0.0003], 'total_ret_net': [np.nan], 'active_ret_net': [np.nan],
    }, index=pd.DatetimeIndex([month], name='month'))

    out = missing_return_sensitivity(holdings, panel, returns, long_fill=-0.30, short_fill=0.30)
    row = out.loc[month]

    # permno 2 is a long (weight 0.4>0) -> filled with long_fill=-0.30
    # permno 4 is a short (weight -0.3<0) -> filled with short_fill=0.30
    expected_long_ret = 0.6 * 0.05 + 0.4 * -0.30
    expected_short_ret = -0.7 * 0.01 + -0.3 * 0.30
    expected_ls = expected_long_ret + expected_short_ret

    assert row['long_ret'] == pytest.approx(expected_long_ret)
    assert row['short_ret'] == pytest.approx(expected_short_ret)
    assert row['ls_ret'] == pytest.approx(expected_ls)
    assert row['total_ret'] == pytest.approx(0.002 + expected_ls)
    assert row['active_ret'] == pytest.approx(row['total_ret'] - 0.005)
    assert row['total_ret_net'] == pytest.approx(row['total_ret'] - 0.0003)
    assert row['active_ret_net'] == pytest.approx(row['active_ret'] - 0.0003)


# ---------------------------------------------------------- smooth
def test_smooth_past_only_and_reset():
    rows = [
        (1, '2020-01-31', 10.0), (2, '2020-01-31', 20.0),
        (1, '2020-02-28', 12.0), (2, '2020-02-28', 8.0),
        (1, '2020-03-31', 14.0), (2, '2020-03-31', 6.0),
        (1, '2020-05-31', 999.0), (2, '2020-05-31', 5.0),  # April skipped for permno 1
    ]
    df = pd.DataFrame(rows, columns=['permno', 'eom', 'x'])
    df['eom'] = pd.to_datetime(df['eom'])

    s = smooth(df, 'x')

    def get(permno, eom):
        return s[(df['permno'] == permno) & (df['eom'] == eom)].iloc[0]

    # hand-computed EMA chain for permno 1 (alpha=0.5): z_jan=-1, z_feb=1, z_mar=1
    assert get(1, '2020-01-31') == pytest.approx(-1.0)
    assert get(1, '2020-02-28') == pytest.approx(0.0)
    assert get(1, '2020-03-31') == pytest.approx(0.5)
    # after the April gap, May resets to its own z-score (no blending with March)
    assert get(1, '2020-05-31') == pytest.approx(1.0)

    # changing month t+1 (March) leaves month t (Feb) unchanged -> past-only
    df2 = df.copy()
    df2.loc[(df2['permno'] == 1) & (df2['eom'] == '2020-03-31'), 'x'] = 500.0
    s2 = smooth(df2, 'x')

    def get2(permno, eom):
        return s2[(df2['permno'] == permno) & (df2['eom'] == eom)].iloc[0]

    assert get2(1, '2020-01-31') == pytest.approx(get(1, '2020-01-31'))
    assert get2(1, '2020-02-28') == pytest.approx(get(1, '2020-02-28'))


# ---------------------------------------------------------- backtest
def test_backtest_tiny_synthetic():
    """3-4 formation months, ~600 stocks: holding month = eom + 1 month-end,
    rf_m/sp500_ret come from the HOLDING month of `market` (not the formation
    month), and a `relax` column is recorded."""
    rng = np.random.default_rng(7)
    n_stocks = 600
    permnos = np.arange(1, n_stocks + 1)
    sector_map = {p: SECTORS[i % len(SECTORS)] for i, p in enumerate(permnos)}
    formation_months = pd.to_datetime(['2020-11-30', '2020-12-31', '2021-01-31', '2021-02-28'])
    holding_months = [m + pd.offsets.MonthEnd(1) for m in formation_months]

    sig_frames, panel_frames = [], []
    for mth in formation_months:
        signal = rng.normal(size=n_stocks)
        sig_frames.append(pd.DataFrame({'permno': permnos, 'eom': mth, 'signal': signal}))
        panel_frames.append(pd.DataFrame({
            'permno': permnos, 'eom': mth,
            'beta': 1.0,  # net(w)=0 -> beta@w trivially within tolerance
            'gics2': [sector_map[p] for p in permnos],
            'size_z': rng.normal(size=n_stocks),
            'ticker': [f'T{p}' for p in permnos],
            'company_name': [f'Co {p}' for p in permnos],
            'stock_exret': rng.normal(0, 0.05, size=n_stocks),
            'has_filing': rng.integers(0, 2, size=n_stocks),
        }))
    signal_df = pd.concat(sig_frames, ignore_index=True)
    panel = pd.concat(panel_frames, ignore_index=True)

    # market keyed ONLY at the HOLDING months, each with a distinctive value. If
    # backtest mistakenly indexed market by the formation month (eom) instead of
    # eom + 1 month-end, the lookup would miss for every month except where the
    # two happen to coincide, which none do here (formation months are all
    # earlier than every holding month).
    market = pd.DataFrame({
        'rf_m': [0.001 + 0.0001 * i for i in range(len(holding_months))],
        'sp500_ret': [0.01 + 0.001 * i for i in range(len(holding_months))],
    }, index=pd.DatetimeIndex(holding_months, name='eom'))
    market['sp500_exret'] = market['sp500_ret'] - market['rf_m']

    holdings, returns = backtest(signal_df, panel, market)

    for mth, hm in zip(formation_months, holding_months):
        sub = holdings.loc[holdings['eom'] == mth]
        assert len(sub) > 0
        assert (sub['month'] == hm).all()

    assert 'relax' in returns.columns
    assert list(returns.index) == holding_months
    for hm, exp_rf, exp_sp in zip(holding_months, market['rf_m'], market['sp500_ret']):
        assert returns.loc[hm, 'rf_m'] == pytest.approx(exp_rf)
        assert returns.loc[hm, 'sp500_ret'] == pytest.approx(exp_sp)


def test_backtest_filer_net_neutral_diagnostic():
    """Panel carries has_filing, strongly correlated with signal: backtest should add a
    filer_net column and keep it within SECTOR_TOL every month (A10)."""
    rng = np.random.default_rng(11)
    n_stocks = 600
    permnos = np.arange(1, n_stocks + 1)
    sector_map = {p: SECTORS[i % len(SECTORS)] for i, p in enumerate(permnos)}
    formation_months = pd.to_datetime(['2020-11-30', '2020-12-31'])
    holding_months = [m + pd.offsets.MonthEnd(1) for m in formation_months]

    sig_frames, panel_frames = [], []
    for mth in formation_months:
        latent = rng.normal(size=n_stocks)
        signal = latent + rng.normal(0, 1.8, size=n_stocks)  # correlated, not deterministic
        sig_frames.append(pd.DataFrame({'permno': permnos, 'eom': mth, 'signal': signal}))
        panel_frames.append(pd.DataFrame({
            'permno': permnos, 'eom': mth, 'beta': 1.0,
            'gics2': [sector_map[p] for p in permnos],
            'size_z': rng.normal(size=n_stocks),
            'ticker': [f'T{p}' for p in permnos],
            'company_name': [f'Co {p}' for p in permnos],
            'stock_exret': rng.normal(0, 0.05, size=n_stocks),
            'has_filing': (latent > 0).astype(int),
        }))
    signal_df = pd.concat(sig_frames, ignore_index=True)
    panel = pd.concat(panel_frames, ignore_index=True)
    market = pd.DataFrame({
        'rf_m': [0.001, 0.0011], 'sp500_ret': [0.01, 0.011],
    }, index=pd.DatetimeIndex(holding_months, name='eom'))
    market['sp500_exret'] = market['sp500_ret'] - market['rf_m']

    holdings, returns = backtest(signal_df, panel, market)
    assert 'filer_net' in returns.columns
    assert (returns['filer_net'].abs() <= config.SECTOR_TOL + 1e-5).all()


def test_backtest_raises_if_holding_month_missing_from_market():
    n_stocks = 600
    permnos = np.arange(1, n_stocks + 1)
    mth = pd.Timestamp('2021-01-31')
    rng = np.random.default_rng(1)
    signal_df = pd.DataFrame({'permno': permnos, 'eom': mth, 'signal': rng.normal(size=n_stocks)})
    panel = pd.DataFrame({
        'permno': permnos, 'eom': mth, 'beta': 1.0, 'gics2': '10', 'size_z': 0.0,
        'ticker': [f'T{p}' for p in permnos], 'company_name': [f'Co {p}' for p in permnos],
        'stock_exret': 0.0, 'has_filing': rng.integers(0, 2, size=n_stocks),
    })
    market = pd.DataFrame({'rf_m': [], 'sp500_ret': [], 'sp500_exret': []},
                           index=pd.DatetimeIndex([], name='eom'))
    with pytest.raises(ValueError):
        backtest(signal_df, panel, market)


def test_backtest_error_names_formation_month_on_infeasible_optimize_month(monkeypatch):
    """When optimize_month raises RuntimeError (genuine infeasibility under the neutrality
    ladder), backtest must re-raise a RuntimeError whose message names the formation month
    that failed -- crucial for diagnosing which ablation signal/month broke, not just that
    something did."""
    n_stocks = 50
    permnos = np.arange(1, n_stocks + 1)
    mth = pd.Timestamp('2021-01-31')
    holding_month = mth + pd.offsets.MonthEnd(1)
    rng = np.random.default_rng(1)
    signal_df = pd.DataFrame({'permno': permnos, 'eom': mth, 'signal': rng.normal(size=n_stocks)})
    panel = pd.DataFrame({
        'permno': permnos, 'eom': mth, 'beta': 1.0, 'gics2': '10', 'size_z': 0.0,
        'ticker': [f'T{p}' for p in permnos], 'company_name': [f'Co {p}' for p in permnos],
        'stock_exret': 0.0, 'has_filing': rng.integers(0, 2, size=n_stocks),
    })
    market = pd.DataFrame({'rf_m': [0.001], 'sp500_ret': [0.01], 'sp500_exret': [0.009]},
                           index=pd.DatetimeIndex([holding_month], name='eom'))

    def fake_optimize_month(*args, **kwargs):
        raise RuntimeError("optimize_month: solver failed even after relaxation")

    monkeypatch.setattr(portfolio, 'optimize_month', fake_optimize_month)

    with pytest.raises(RuntimeError, match=str(mth)):
        backtest(signal_df, panel, market)


# ---------------------------------------------------------- labels
def test_attach_labels_never_uses_future_dates():
    holdings = pd.DataFrame({
        'month': pd.to_datetime(['2021-07-31'] * 3),
        'eom': pd.to_datetime(['2021-06-30'] * 3),
        'permno': [1, 2, 3],
        'weight': [0.01, -0.01, 0.01],
        'ticker': [None, None, None],
        'company_name': [None, None, None],
        'label_source': [None, None, None],
    })
    label_panel = pd.DataFrame({
        'permno': [1, 1],
        'eom': pd.to_datetime(['2021-05-31', '2021-07-31']),  # second is AFTER eom
        'ticker': ['OLD', 'NEW'],
        'company_name': ['Old Co', 'New Co'],
    })
    filing_labels = pd.DataFrame({
        'permno': [2, 2],
        'filing_date': pd.to_datetime(['2021-04-15', '2021-08-01']),  # second is AFTER eom
        'ticker': ['F2', 'FUTURE'],
        'company_name': ['Filing Co', 'Future Co'],
    })

    out = attach_labels(holdings, label_panel, filing_labels)
    out = out.set_index('permno')

    assert out.loc[1, 'ticker'] == 'OLD'
    assert out.loc[1, 'label_source'] == 'raw_panel'
    assert out.loc[2, 'ticker'] == 'F2'
    assert out.loc[2, 'label_source'] == 'filing'
    assert out.loc[3, 'label_source'] == 'UNLABELED'
    assert out.loc[3, 'ticker'] == 'UNLABELED'


def test_attach_labels_mixed_datetime_resolution():
    """merge_asof (via _asof_fill) requires identical merge-key dtypes; on newer pandas builds
    a holdings frame and a label source can each carry a different, non-ns datetime resolution
    (e.g. one parquet/csv round-trip reads back as datetime64[ms], another as datetime64[s]).
    attach_labels must not crash and must still resolve labels correctly."""
    holdings = pd.DataFrame({
        'month': pd.to_datetime(['2021-07-31']),
        'eom': pd.to_datetime(['2021-06-30']).astype('datetime64[ms]'),
        'permno': [1],
        'weight': [0.01],
        'ticker': [None],
        'company_name': [None],
        'label_source': [None],
    })
    label_panel = pd.DataFrame({
        'permno': [1],
        'eom': pd.to_datetime(['2021-05-31']).astype('datetime64[s]'),
        'ticker': ['OLD'],
        'company_name': ['Old Co'],
    })
    filing_labels = pd.DataFrame({
        'permno': pd.Series([], dtype='int64'),
        'filing_date': pd.Series([], dtype='datetime64[s]'),
        'ticker': pd.Series([], dtype='object'),
        'company_name': pd.Series([], dtype='object'),
    })

    out = attach_labels(holdings, label_panel, filing_labels).set_index('permno')

    assert out.loc[1, 'ticker'] == 'OLD'
    assert out.loc[1, 'label_source'] == 'raw_panel'


# ---------------------------------------------------------- write_submission
def test_write_submission_format_and_rounding(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SUB_DIR', tmp_path)

    n_long, n_short = 70, 70  # enough names at MAX_WEIGHT=1.5% to reach a 100% leg
    permno = np.arange(1, n_long + n_short + 1)
    weight = np.concatenate([np.full(n_long, 1.0 / n_long), -np.full(n_short, 1.0 / n_short)])

    holdings = pd.DataFrame({
        'month': pd.Timestamp('2021-02-28'), 'eom': pd.Timestamp('2021-01-31'),
        'permno': permno, 'weight': weight,
        'ticker': [f'T{p}' for p in permno], 'company_name': [f'Co {p}' for p in permno],
        'label_source': 'panel',
    })
    returns = pd.DataFrame({
        'total_ret': [0.01], 'rf_m': [0.002], 'bench_ret': [0.005], 'active_ret': [0.005],
        'ls_ret': [0.008], 'long_ret': [0.006], 'short_ret': [0.002], 'sp500_ret': [0.02],
        'total_ret_net': [0.0098], 'active_ret_net': [0.0048],
    }, index=pd.DatetimeIndex(['2021-02-28'], name='month'))

    write_submission(holdings, returns)

    h = pd.read_csv(tmp_path / 'holdings.csv')
    assert list(h.columns) == ['Date', 'PERMNO', 'TICKER', 'COMPANY NAME', 'WEIGHT']
    assert h['Date'].iloc[0] == '2021-02-01'
    assert h['WEIGHT'].abs().max() <= config.MAX_WEIGHT * 100.0 + 1e-9
    assert h.loc[h['WEIGHT'] > 0, 'WEIGHT'].sum() == pytest.approx(100.0, abs=1e-6)
    assert h.loc[h['WEIGHT'] < 0, 'WEIGHT'].sum() == pytest.approx(-100.0, abs=1e-6)

    r = pd.read_csv(tmp_path / 'returns.csv')
    assert list(r.columns) == ['Date', 'total_ret', 'rf_m', 'bench_ret', 'active_ret',
                                'ls_ret', 'long_ret', 'short_ret', 'sp500_ret',
                                'total_ret_net', 'active_ret_net']
    assert r['Date'].iloc[0] == '2021-02-01'
    assert r['total_ret'].iloc[0] == pytest.approx(0.01)

    audit = pd.read_csv(tmp_path / 'label_audit.csv')
    assert set(['permno', 'month', 'ticker', 'company_name', 'label_source']).issubset(audit.columns)
    assert audit['month'].iloc[0] == '2021-02-01'  # first-of-holding-month, like holdings.csv


def test_write_submission_rejects_null_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SUB_DIR', tmp_path)
    holdings = pd.DataFrame({
        'month': pd.to_datetime(['2021-02-28']), 'eom': pd.to_datetime(['2021-01-31']),
        'permno': [1], 'weight': [0.01], 'ticker': [None], 'company_name': ['A Co'],
        'label_source': ['panel'],
    })
    returns = pd.DataFrame({
        'total_ret': [0.01], 'rf_m': [0.002], 'bench_ret': [0.005], 'active_ret': [0.005],
        'ls_ret': [0.008], 'long_ret': [0.006], 'short_ret': [0.002], 'sp500_ret': [0.02],
        'total_ret_net': [0.0098], 'active_ret_net': [0.0048],
    }, index=pd.DatetimeIndex(['2021-02-28'], name='month'))
    with pytest.raises(AssertionError):
        write_submission(holdings, returns)
