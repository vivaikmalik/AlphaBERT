"""Tests for src/beta.py (point-in-time stock-beta estimates + uncertainty, docs/research_log.md
A15/A16). Fast, synthetic-data only -- no real data or network access."""
import numpy as np
import pandas as pd
import pytest

from src import beta as beta_mod
from src import data as data_mod


# ---------------------------------------------------------------------------
# synthetic multi-year panel: many permnos with a known constant true beta, so
# forward_beta / fit_beta_params can be tested against something we control.
# ---------------------------------------------------------------------------

YOUNG_PERMNOS = list(range(5))          # always missing beta_60m, like real young stocks
YOUNG_TRUE_BETA = 1.6                   # mirrors docs/research_log.md's ~1.25-1.7 finding


def _synthetic_calib_raw(n_months=60, n_permnos=40, start='2015-01-31', seed=0):
    """Panel spanning `start`..`start`+n_months-1 (default 2015-01..2019-12) with a known
    per-permno true beta. Returns (raw, mkt_ret) where mkt_ret is a plain Series indexed by eom
    that reproduces exactly what market_state() would need to be monkeypatched to return, so
    forward_beta recovers each permno's true_beta up to noise."""
    rng = np.random.default_rng(seed)
    eoms = pd.date_range(start, periods=n_months, freq='ME')
    mkt_ret = pd.Series(rng.normal(0.01, 0.04, n_months), index=eoms, name='mkt_ret')

    permnos = np.arange(1000, 1000 + n_permnos)
    true_beta = {p: (YOUNG_TRUE_BETA if i in YOUNG_PERMNOS else rng.uniform(0.4, 1.6))
                 for i, p in enumerate(permnos)}
    # keep the cross-sectional mean near 1.0 so the synthetic "market" is roughly self-consistent
    gics_pool = ['10101010', '20202020', '30303030']

    rows = []
    for eom in eoms:
        for i, p in enumerate(permnos):
            tb = true_beta[p]
            idio = rng.normal(0, 0.04)
            ret = tb * mkt_ret.loc[eom] + idio
            is_young = i in YOUNG_PERMNOS
            row = {
                'permno': p, 'eom': eom, 'ret': ret,
                # constant prc/me (above config.MIN_PRICE / any plausible ME_CUTOFF_PCTILE
                # threshold): fit_beta_params/held_out_report now apply data.universe_mask
                # (A16 audit fix #1/#2), and this fixture must not let that filter systematically
                # drop YOUNG_PERMNOS out of the universe.
                'prc': 20.0, 'me': 1e6, 'gics': gics_pool[i % 3],
                'age': 12 + i * 3, 'at_be': 1.5 + 0.05 * i,
                'ivol_capm_252d': abs(rng.normal(0.02, 0.01)),
                'beta_60m': np.nan if is_young else tb + rng.normal(0, 0.15),
                'betabab_1260d': tb + rng.normal(0, 0.15),
                'betadown_252d': tb + rng.normal(0, 0.2),
                'beta_dimson_21d': tb + rng.normal(0, 0.3),
            }
            rows.append(row)
    raw = pd.DataFrame(rows)
    return raw, mkt_ret, true_beta


@pytest.fixture()
def synth(tmp_path, monkeypatch):
    from src import config
    raw, mkt_ret, true_beta = _synthetic_calib_raw()

    def fake_market_state():
        return pd.DataFrame({'mkt_ret': mkt_ret})

    monkeypatch.setattr(data_mod, 'market_state', fake_market_state)
    # fit_beta_params() writes CACHE_DIR/'beta_params.json' as a side effect -- must never touch
    # the real cache (would silently overwrite real calibration numbers with synthetic ones).
    monkeypatch.setattr(config, 'CACHE_DIR', tmp_path)
    return raw, mkt_ret, true_beta


# --------------------------------------------------------------------- forward_beta ---
def test_forward_beta_recovers_known_beta_no_noise():
    """With an exact linear relationship (no idio noise), forward_beta must recover the true
    beta almost exactly for a formation month with a full forward window."""
    eoms = pd.date_range('2015-01-31', periods=14, freq='ME')
    mkt = pd.Series(np.linspace(-0.05, 0.05, 14), index=eoms)
    rows = []
    true_beta = {1: 0.5, 2: 1.5, 3: 2.2}
    for eom in eoms:
        for p, b in true_beta.items():
            rows.append({'permno': p, 'eom': eom, 'ret': b * mkt.loc[eom]})
    raw = pd.DataFrame(rows)

    fwd = beta_mod.forward_beta(raw, mkt)
    t0 = eoms[0]
    for p, b in true_beta.items():
        assert fwd.loc[(p, t0)] == pytest.approx(b, abs=1e-6)


def test_forward_beta_requires_full_window():
    """A stock missing even one month within the forward 12-month window gets no forward beta at
    that formation month (dropped, not interpolated)."""
    eoms = pd.date_range('2015-01-31', periods=13, freq='ME')
    mkt = pd.Series(np.linspace(-0.05, 0.05, 13), index=eoms)
    rows = []
    for i, eom in enumerate(eoms):
        if i == 6:  # gap in the middle of permno 1's forward window from t0
            continue
        rows.append({'permno': 1, 'eom': eom, 'ret': 1.0 * mkt.loc[eom]})
    raw = pd.DataFrame(rows)
    fwd = beta_mod.forward_beta(raw, mkt)
    assert (1, eoms[0]) not in fwd.index


# --------------------------------------------------------------------- fit_beta_params (point-in-time) ---
def test_fit_beta_params_asserts_formation_range(synth):
    raw, _, _ = synth
    params = beta_mod.fit_beta_params(raw)
    assert params['calib_start'] == '2015-01-31'
    assert params['calib_end'] == '2017-12-31'
    # sanity: every raw estimator got a calibration and the prior got fitted
    assert set(params['estimators']) == set(beta_mod.RAW_ESTIMATORS)
    assert 'Vp' in params['prior'] and params['prior']['Vp'] > 0


def test_fit_beta_params_point_in_time(synth):
    """Mutating data dated 2019-01 or later must not change any fitted parameter: fit_beta_params
    only uses formation (regressor) rows eom<=2017-12, and its forward-beta TARGET only needs
    returns through 2018-12 (the last calibration formation month's forward window)."""
    raw, mkt_ret, _ = synth
    params_a = beta_mod.fit_beta_params(raw)

    raw2 = raw.copy()
    post_2019 = raw2['eom'] >= pd.Timestamp('2019-01-31')
    assert post_2019.sum() > 0
    rng = np.random.default_rng(999)
    n = post_2019.sum()
    raw2.loc[post_2019, 'ret'] = rng.normal(0, 1, n)
    for j in beta_mod.RAW_ESTIMATORS:
        raw2.loc[post_2019, j] = rng.normal(0, 5, n)

    params_b = beta_mod.fit_beta_params(raw2)

    import json
    assert json.dumps(params_a, sort_keys=True) == json.dumps(params_b, sort_keys=True)


def test_fit_beta_params_formation_data_does_matter(synth):
    """Sanity counterpart to the point-in-time test: mutating pre-2018 (formation-window) data
    DOES change the fitted params, proving the point-in-time test isn't vacuously true."""
    raw, mkt_ret, _ = synth
    params_a = beta_mod.fit_beta_params(raw)

    raw2 = raw.copy()
    pre_2018 = raw2['eom'] <= pd.Timestamp('2017-12-31')
    rng = np.random.default_rng(123)
    raw2.loc[pre_2018, 'beta_60m'] = rng.normal(0, 5, pre_2018.sum())
    params_b = beta_mod.fit_beta_params(raw2)

    assert params_a['estimators']['beta_60m'] != params_b['estimators']['beta_60m']


# --------------------------------------------------------------------- fusion ---
def test_fusion_missing_estimators_falls_back_to_prior(synth, monkeypatch):
    raw, _, _ = synth
    params = beta_mod.fit_beta_params(raw)
    monkeypatch.setattr(beta_mod, '_load_or_fit_params', lambda: params)

    df = pd.DataFrame({
        'permno': [9999], 'eom': [pd.Timestamp('2018-06-30')],
        'me': [5e6], 'gics': ['10101010'], 'age': [40.0], 'at_be': [1.8],
        'ivol_capm_252d': [0.03],
        'beta_60m': [np.nan], 'betabab_1260d': [np.nan],
        'betadown_252d': [np.nan], 'beta_dimson_21d': [np.nan],
    })
    out = beta_mod.compute_betas(df, 'fusion')

    prior_mean, prior_var = beta_mod._prior_mean_var(df, params)
    assert out['beta'].iloc[0] == pytest.approx(prior_mean.iloc[0])
    assert out['beta_var'].iloc[0] == pytest.approx(prior_var)


def test_fusion_variance_shrinks_as_estimators_added(synth, monkeypatch):
    raw, _, _ = synth
    params = beta_mod.fit_beta_params(raw)
    monkeypatch.setattr(beta_mod, '_load_or_fit_params', lambda: params)

    base = {'permno': 1, 'eom': pd.Timestamp('2018-06-30'), 'me': 5e6, 'gics': '10101010',
            'age': 40.0, 'at_be': 1.8, 'ivol_capm_252d': 0.03}
    row0 = {**base, 'beta_60m': np.nan, 'betabab_1260d': np.nan,
            'betadown_252d': np.nan, 'beta_dimson_21d': np.nan}
    row1 = {**row0, 'beta_60m': 1.1}
    row2 = {**row1, 'betabab_1260d': 1.2}
    row3 = {**row2, 'betadown_252d': 1.0, 'beta_dimson_21d': 1.3}
    df = pd.DataFrame([row0, row1, row2, row3])

    out = beta_mod.compute_betas(df, 'fusion')
    var = out['beta_var'].to_numpy()
    assert var[0] > var[1] > var[2] > var[3]


# --------------------------------------------------------------------- kalman ---
def test_kalman_tracks_drifting_beta():
    """A single permno's beta drifts linearly over 24 months; noisy monthly measurements (one
    calibrated estimator, small noise) should let the Kalman filter track the drift far better
    than the flat prior alone."""
    rng = np.random.default_rng(7)
    n = 24
    eoms = pd.date_range('2015-01-31', periods=n, freq='ME')
    true_beta = np.linspace(0.5, 2.0, n)

    df = pd.DataFrame({'permno': [1] * n, 'eom': eoms})
    prior_mean = pd.Series(1.0, index=df.index)  # flat, uninformative prior
    prior_var = 1.0
    noisy_meas = true_beta + rng.normal(0, 0.05, n)
    cal = {'beta_60m': (pd.Series(noisy_meas, index=df.index), 0.01)}  # small R -> trusted
    params = {'kalman': {'phi': 0.95, 'q': 0.05}}

    filt_mean, filt_var = beta_mod._kalman(df, prior_mean, prior_var, cal, params)

    mae_kalman = np.mean(np.abs(filt_mean.to_numpy() - true_beta))
    mae_prior_only = np.mean(np.abs(prior_mean.to_numpy() - true_beta))
    assert mae_kalman < 0.15
    assert mae_kalman < 0.3 * mae_prior_only
    assert np.corrcoef(filt_mean.to_numpy(), true_beta)[0, 1] > 0.9
    assert (filt_var > 0).all()


def test_kalman_multiple_permnos_independent():
    """Two permnos processed in the same call must not leak state into each other (verified by
    checking each series independently matches what a solo single-permno run would give)."""
    rng = np.random.default_rng(11)
    n = 10
    eoms = pd.date_range('2015-01-31', periods=n, freq='ME')
    params = {'kalman': {'phi': 0.9, 'q': 0.02}}

    def run(permnos_meas):
        dfs, prior_means, cals = [], [], []
        for permno, meas in permnos_meas.items():
            sub = pd.DataFrame({'permno': [permno] * n, 'eom': eoms})
            dfs.append(sub)
        df = pd.concat(dfs, ignore_index=True)
        prior_mean = pd.Series(1.0, index=df.index)
        meas_all = np.concatenate(list(permnos_meas.values()))
        cal = {'beta_60m': (pd.Series(meas_all, index=df.index), 0.02)}
        return beta_mod._kalman(df, prior_mean, 1.0, cal, params)

    meas_a = 1.2 + rng.normal(0, 0.05, n)
    meas_b = 0.7 + rng.normal(0, 0.05, n)
    combined_mean, _ = run({1: meas_a, 2: meas_b})

    solo_a_mean, _ = run({1: meas_a})
    solo_b_mean, _ = run({2: meas_b})

    combined_a = combined_mean.iloc[:n].to_numpy()
    combined_b = combined_mean.iloc[n:].to_numpy()
    assert np.allclose(combined_a, solo_a_mean.to_numpy())
    assert np.allclose(combined_b, solo_b_mean.to_numpy())


# --------------------------------------------------------------------- blume / a15 unchanged ---
def test_compute_betas_blume_matches_formula():
    from src import config
    df = pd.DataFrame({
        'permno': [1, 2, 3], 'eom': pd.Timestamp('2020-01-31'),
        'beta_60m': [1.2, np.nan, 0.4],
        'betabab_1260d': [9.0, 9.0, 9.0], 'ivol_capm_252d': [0.5, 0.5, 0.5],
        'betadown_252d': [0.0, 0.0, 0.0], 'beta_dimson_21d': [0.0, 0.0, 0.0],
        'me': [1e6, 1e6, 1e6], 'gics': ['10101010'] * 3, 'age': [50, 50, 50], 'at_be': [1.5] * 3,
    })
    out = beta_mod.compute_betas(df, 'blume')
    expected = ((1 - config.BETA_SHRINK) * df['beta_60m'] + config.BETA_SHRINK).fillna(1.0)
    assert np.allclose(out['beta'], expected)
    assert (out['beta_var'] == 0.0).all()


def test_compute_betas_a15_matches_formula():
    from src import config
    df = pd.DataFrame({
        'permno': [1, 2, 3], 'eom': pd.Timestamp('2020-01-31'),
        'beta_60m': [1.2, 1.5, np.nan],
        'betabab_1260d': [1.0, np.nan, np.nan], 'ivol_capm_252d': [0.1, 0.5, np.nan],
        'betadown_252d': [0.0, 0.0, 0.0], 'beta_dimson_21d': [0.0, 0.0, 0.0],
        'me': [1e6, 1e6, 1e6], 'gics': ['10101010'] * 3, 'age': [50, 50, 50], 'at_be': [1.5] * 3,
    })
    out = beta_mod.compute_betas(df, 'a15')

    b1 = ((config.BETA_FP_W * df['betabab_1260d'].clip(-1, 4) + config.BETA_FP_C)
          .fillna(0.67 * df['beta_60m'] + 0.33).fillna(config.BETA_MISSING))
    ivp = df['ivol_capm_252d'].groupby(df['eom']).rank(pct=True).fillna(0.5)
    expected = config.BETA_INTERCEPT + config.BETA_SLOPE * b1 + config.BETA_IVOL * ivp
    assert np.allclose(out['beta'], expected)
    assert (out['beta_var'] == 0.0).all()


# --------------------------------------------------------------------- integration: the actual fix ---
def test_fusion_beats_blume_for_young_stocks(synth, monkeypatch):
    """The whole point of this module (docs/research_log.md): young stocks missing beta_60m get
    imputed to a flat 1.0 under 'blume' even though their true forward beta is much higher
    (~1.6 here). 'fusion' should get noticeably closer."""
    raw, _, true_beta = synth
    params = beta_mod.fit_beta_params(raw)
    monkeypatch.setattr(beta_mod, '_load_or_fit_params', lambda: params)

    young_permnos = [1000 + i for i in YOUNG_PERMNOS]
    holdout = raw.loc[(raw['eom'] == pd.Timestamp('2018-06-30')) & raw['permno'].isin(young_permnos)]
    assert len(holdout) == len(young_permnos)

    blume = beta_mod.compute_betas(holdout, 'blume')['beta']
    fusion = beta_mod.compute_betas(holdout, 'fusion')['beta']

    assert (blume == 1.0).all()  # the bug, reproduced: flat imputation
    err_blume = (blume - YOUNG_TRUE_BETA).abs().mean()
    err_fusion = (fusion - YOUNG_TRUE_BETA).abs().mean()
    assert err_fusion < err_blume


def test_held_out_report_smoke(synth, monkeypatch):
    raw, _, _ = synth
    params = beta_mod.fit_beta_params(raw)
    monkeypatch.setattr(beta_mod, '_load_or_fit_params', lambda: params)
    report = beta_mod.held_out_report(raw)
    assert set(report['model']) == {'blume', 'a15', 'fusion', 'kalman'}
    assert set(report['subset']) == {'all', 'missing_beta_60m'}
    assert (report['n'] > 0).all()
    assert report[['r2', 'mean_bias']].notna().all().all()
