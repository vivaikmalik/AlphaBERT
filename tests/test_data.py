import numpy as np
import pandas as pd
import pytest

from src import config
from src import data as data_mod
from src.data import (
    _rank_within_eom, universe_mask, build_panel, feature_columns, market_state, load_market,
    load_chars, pipeline_rf,
)


# ---------------------------------------------------------------------------
# fast unit tests on synthetic data
# ---------------------------------------------------------------------------

def test_rank_mapping_range_ties_nan():
    df = pd.DataFrame({
        'eom': ['2020-01-31'] * 5 + ['2020-02-28'] * 3,
        'x': [1.0, 2.0, 2.0, 3.0, np.nan, 5.0, 5.0, np.nan],
    })
    df['eom'] = pd.to_datetime(df['eom'])
    out = _rank_within_eom(df, ['x'])

    # range within [-1, 1]
    assert out['x'].between(-1, 1).all()

    # NaN input -> 0 output
    assert out.loc[df['x'].isna(), 'x'].eq(0.0).all()

    # group 1: values 1,2,2,3 (n=4 non-null); average-tie rank of the two 2.0s is 2.5
    # scaled = 2*(rank-1)/(n-1) - 1
    g1 = out.loc[df['eom'] == '2020-01-31', 'x'].tolist()
    expected_1 = 2 * (1 - 1) / 3 - 1       # rank 1 -> value 1.0
    expected_2 = 2 * (2.5 - 1) / 3 - 1     # rank 2.5 -> the two tied 2.0s
    expected_3 = 2 * (4 - 1) / 3 - 1       # rank 4 -> value 3.0
    assert g1[0] == pytest.approx(expected_1)
    assert g1[1] == pytest.approx(expected_2)
    assert g1[2] == pytest.approx(expected_2)
    assert g1[3] == pytest.approx(expected_3)
    assert g1[4] == 0.0

    # min/max of a group with all non-null values should hit -1 and +1
    assert g1[0] == pytest.approx(-1.0)
    assert g1[3] == pytest.approx(1.0)

    # group 2: 5, 5, NaN -> n=2 non-null, tied at rank 1.5 each
    g2 = out.loc[df['eom'] == '2020-02-28', 'x'].tolist()
    expected_tied = 2 * (1.5 - 1) / (2 - 1) - 1
    assert g2[0] == pytest.approx(expected_tied)
    assert g2[1] == pytest.approx(expected_tied)
    assert g2[2] == 0.0


def test_rank_mapping_n_equals_1_is_zero():
    df = pd.DataFrame({'eom': pd.to_datetime(['2020-01-31']), 'x': [42.0]})
    out = _rank_within_eom(df, ['x'])
    assert out['x'].iloc[0] == 0.0


def _synthetic_raw(n_months=3, n_permnos=6, seed=0):
    """Small raw frame with all 147 char columns + id/aux columns, for build_panel tests."""
    rng = np.random.default_rng(seed)
    chars = config.load_char_list()
    eoms = pd.date_range('2016-01-31', periods=n_months, freq='ME')
    rows = []
    for m_i, eom in enumerate(eoms):
        for p in range(n_permnos):
            permno = 1000 + p
            row = {
                'permno': permno,
                'eom': eom,
                'date': eom,
                'prc': 10.0 + p + m_i,           # all >= MIN_PRICE(5)
                'me': 100.0 * (p + 1),
                'gics': None if p == 0 else f'{10 + p}101010',
                'beta_60m': np.nan if p == 1 else 1.0 + 0.1 * p,
                'dolvol_126d': 1000.0 * (p + 1),
                'size_grp': 'micro' if p < 3 else 'mega',
                'ticker': f'T{p}',
                'company_name': f'Company {p}',
                # terminal month has no next-month return for a couple of permnos
                'ret_exc_lead1m': np.nan if (m_i == n_months - 1 and p < 2) else rng.normal(0, 0.05),
            }
            for c in chars:
                v = rng.normal(0, 1)
                # sprinkle NaNs on a couple of chars to exercise miss_ flags / NaN->0 rank
                if c in ('age', 'f_score') and rng.random() < 0.4:
                    v = np.nan
                row[c] = v
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_env(tmp_path, monkeypatch):
    raw = _synthetic_raw()
    chars_path = tmp_path / 'synthetic_chars.parquet'
    raw.to_parquet(chars_path)
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'CHARS_PATH', chars_path)
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)
    # the synthetic file lacks the extra raw beta columns the 'fusion'/'kalman' models fit on;
    # these panel tests are not about beta, so pin the simple model
    monkeypatch.setattr(config, 'BETA_MODEL', 'blume')
    return raw


def test_universe_mask_ignores_ret_exc_lead1m(synthetic_env):
    raw = synthetic_env
    mask1 = universe_mask(raw)

    raw2 = raw.copy()
    raw2['ret_exc_lead1m'] = np.nan
    mask2 = universe_mask(raw2)
    assert (mask1 == mask2).all()

    raw3 = raw.copy()
    rng = np.random.default_rng(1)
    raw3['ret_exc_lead1m'] = rng.normal(size=len(raw3))
    mask3 = universe_mask(raw3)
    assert (mask1 == mask3).all()


def test_target_month_and_stock_exret(synthetic_env):
    panel = build_panel()
    expected_target = panel['eom'] + pd.offsets.MonthEnd(1)
    assert (panel['target_month'] == expected_target).all()

    # stock_exret is exactly ret_exc_lead1m, never shifted again: recover it from the raw frame
    raw = synthetic_env.set_index(['permno', 'eom'])
    merged = panel.set_index(['permno', 'eom']).join(raw[['ret_exc_lead1m']], rsuffix='_raw')
    both_null = merged['stock_exret'].isna() & merged['ret_exc_lead1m'].isna()
    close = np.isclose(merged['stock_exret'], merged['ret_exc_lead1m'], equal_nan=True)
    assert (close | both_null).all()


def test_feature_columns_excludes_target_derived(synthetic_env):
    panel = build_panel()
    feats = feature_columns(panel)
    banned = {'stock_exret', 'ret_exc_lead1m', 'target_month', 'eom', 'date', 'permno'}
    assert banned.isdisjoint(feats)
    assert len(feats) == len(set(feats))


def test_pipeline_rf_recovers_rf_from_ret_minus_ret_exc(tmp_path, monkeypatch):
    """pipeline_rf() = median(ret - ret_exc) per eom -- a hand-computed check with a known,
    per-eom rf plus per-row noise in ret/ret_exc (the rf itself cancels out of that noise since
    it's added identically to both, so the difference is exact regardless of noise)."""
    rng = np.random.default_rng(3)
    rows = []
    known_rf = {pd.Timestamp('2016-01-31'): 0.0011, pd.Timestamp('2016-02-29'): 0.0034}
    for eom, rf in known_rf.items():
        for p in range(5):
            base_ret = rng.normal(0.01, 0.05)
            rows.append({'eom': eom, 'ret': base_ret, 'ret_exc': base_ret - rf})
    raw = pd.DataFrame(rows)
    chars_path = tmp_path / 'rf_chars.parquet'
    raw.to_parquet(chars_path)
    monkeypatch.setattr(config, 'CHARS_PATH', chars_path)

    rf = pipeline_rf()
    assert rf.index.name == 'eom'
    for eom, expected in known_rf.items():
        assert rf.loc[eom] == pytest.approx(expected, abs=1e-12)


def test_pipeline_rf_median_robust_to_a_single_outlier_row(tmp_path, monkeypatch):
    """One row with a corrupted ret_exc (e.g. a stray NaN turned into a bad number) must not
    move the recovered rf -- median, not mean."""
    eom = pd.Timestamp('2016-01-31')
    rf_true = 0.002
    rets = [0.01, 0.02, -0.01, 0.03, 0.00]
    rows = [{'eom': eom, 'ret': r, 'ret_exc': r - rf_true} for r in rets]
    rows.append({'eom': eom, 'ret': 0.01, 'ret_exc': 0.01 - 5.0})  # wild outlier
    raw = pd.DataFrame(rows)
    chars_path = tmp_path / 'rf_chars_outlier.parquet'
    raw.to_parquet(chars_path)
    monkeypatch.setattr(config, 'CHARS_PATH', chars_path)

    rf = pipeline_rf()
    assert rf.loc[eom] == pytest.approx(rf_true, abs=1e-12)


@pytest.mark.slow
def test_real_pipeline_rf_coverage():
    """Real data: pipeline_rf covers the full backtest holding-month window with no NaN, and
    values sit in a plausible short-rate range (well under 5%/month)."""
    rf = pipeline_rf()
    window = rf.loc['2020-12-31':'2026-08-31']
    expected_months = pd.date_range('2020-12-31', '2026-08-31', freq='ME')
    missing = expected_months.difference(window.index)
    assert len(missing) == 0, f'missing months: {missing.tolist()}'
    assert not window.isna().any()
    assert window.abs().max() < 0.05


def test_market_state_truncation_invariance(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    n_months = 15
    n_permnos = 8
    eoms = pd.date_range('2016-01-31', periods=n_months, freq='ME')
    rows = []
    for eom in eoms:
        for p in range(n_permnos):
            rows.append({
                'permno': 2000 + p,
                'eom': eom,
                'prc': 10.0 + p,
                'me': 100.0 * (p + 1),
                'me_lag1': 95.0 * (p + 1),
                'ret': rng.normal(0.01, 0.05),
                'ivol_capm_21d': abs(rng.normal(0.02, 0.01)),
            })
    raw = pd.DataFrame(rows)

    T = eoms[9]  # truncate after this month

    full_path = tmp_path / 'full.parquet'
    raw.to_parquet(full_path)
    monkeypatch.setattr(config, 'CHARS_PATH', full_path)
    full_state = market_state()

    trunc_path = tmp_path / 'trunc.parquet'
    raw.loc[raw['eom'] <= T].to_parquet(trunc_path)
    monkeypatch.setattr(config, 'CHARS_PATH', trunc_path)
    trunc_state = market_state()

    pd.testing.assert_series_equal(full_state.loc[T], trunc_state.loc[T], check_names=False)


def test_miss_flags_use_only_cutoff_rows(tmp_path, monkeypatch):
    """A char that is only missing in months AFTER MISS_FLAG_CUTOFF must NOT be flagged: flag
    selection uses only universe rows with eom <= MISS_FLAG_CUTOFF."""
    chars = config.load_char_list()
    target_char = chars[0]
    rng = np.random.default_rng(3)
    eoms = [config.MISS_FLAG_CUTOFF - pd.offsets.MonthEnd(1), config.MISS_FLAG_CUTOFF,
            config.MISS_FLAG_CUTOFF + pd.offsets.MonthEnd(1), config.MISS_FLAG_CUTOFF + pd.offsets.MonthEnd(2)]
    n_permnos = 10
    rows = []
    for eom in eoms:
        after_cutoff = eom > config.MISS_FLAG_CUTOFF
        for p in range(n_permnos):
            row = {}
            for c in chars:
                # after the cutoff, most universe-eligible rows (p>=2 pass the me filter) go
                # missing for target_char; before/at the cutoff it is never missing.
                if c == target_char and after_cutoff and 2 <= p <= 7:
                    v = np.nan
                else:
                    v = rng.normal(0, 1)
                row[c] = v
            # id/aux columns set AFTER the chars loop: some chars share names with id/aux columns
            # ('prc', 'dolvol_126d', 'beta_60m') and must not be clobbered by the random fill above.
            row.update({
                'permno': 3000 + p, 'eom': eom, 'date': eom,
                'prc': 20.0, 'me': 1000.0 * (p + 1),
                'gics': f'{10 + p}101010', 'beta_60m': 1.0,
                'dolvol_126d': 1000.0, 'size_grp': 'mega',
                'ticker': f'T{p}', 'company_name': f'C{p}',
                'ret_exc_lead1m': rng.normal(0, 0.05),
            })
            rows.append(row)
    raw = pd.DataFrame(rows)

    chars_path = tmp_path / 'chars.parquet'
    raw.to_parquet(chars_path)
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'CHARS_PATH', chars_path)
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)
    monkeypatch.setattr(config, 'BETA_MODEL', 'blume')  # synthetic file lacks fusion's raw beta columns

    panel = build_panel()
    assert f'miss_{target_char}' not in panel.columns


def test_beta_a15_formula_and_gics2_na(tmp_path, monkeypatch):
    """A15 beta formula (config.BETA_MODEL == 'a15', an ablation-only path since A16 reverted the
    default to 'blume'): b1 = BETA_FP_W*clip(betabab_1260d,-1,4) + BETA_FP_C (incl. the clip
    biting on out-of-range betabab), falling back to a Blume-adjusted beta_60m (0.67*beta_60m +
    0.33) when betabab_1260d is missing, and to BETA_MISSING when both are missing; blended with
    ivp = within-eom percentile rank of ivol_capm_252d (0.5 when ivol_capm_252d is itself missing)
    as beta = BETA_INTERCEPT + BETA_SLOPE*b1 + BETA_IVOL*ivp. Also covers gics2 'NA' fill for
    missing gics (unrelated to the beta formula, kept from the pre-A15 version of this test)."""
    monkeypatch.setattr(config, 'BETA_MODEL', 'a15')
    chars = config.load_char_list()
    rng = np.random.default_rng(4)
    eom = pd.Timestamp('2020-01-31')
    # (betabab_1260d, beta_60m, ivol_capm_252d) per permno -- ivol values are strictly increasing
    # so each row's percentile rank among the 7 non-null ivol values is unambiguous (rank/7).
    specs = [
        (1.0, 1.2, 0.01),    # p0: betabab present, in-range -> b1 = 0.6*1.0 + 0.4
        (10.0, 1.2, 0.02),   # p1: betabab present, clipped from 10.0 down to 4
        (-5.0, 1.2, 0.03),   # p2: betabab present, clipped from -5.0 up to -1
        (np.nan, 1.5, 0.04), # p3: betabab missing -> Blume fallback on beta_60m
        (np.nan, np.nan, 0.05),  # p4: both missing -> BETA_MISSING
        (1.0, 1.2, 0.06),    # p5: gics missing (separate check), betabab present
        (2.0, 1.2, 0.07),    # p6: betabab present, in-range
        (0.0, 1.2, np.nan),  # p7: ivol_capm_252d missing -> ivp fills to 0.5
    ]
    rows = []
    for p, (betabab, beta60m, ivol252) in enumerate(specs):
        row = {c: rng.normal(0, 1) for c in chars}
        # id/aux columns set AFTER the chars fill: some chars share names with id/aux columns
        # ('prc', 'dolvol_126d', 'beta_60m', 'betabab_1260d', 'ivol_capm_252d') and must not be
        # clobbered by the random fill above.
        row.update({
            'permno': 4000 + p, 'eom': eom, 'date': eom,
            # constant 'me' (not scaled by p): universe_mask's within-eom ME_CUTOFF_PCTILE=0.20
            # quantile filter would otherwise drop the bottom ~2 of 8 rows at a linearly-spaced
            # 'me', and every one of these 8 rows needs to survive into the panel.
            'prc': 20.0, 'me': 1_000_000.0,
            'gics': None if p == 5 else f'{10 + p}101010',
            'beta_60m': beta60m, 'betabab_1260d': betabab, 'ivol_capm_252d': ivol252,
            'dolvol_126d': 1000.0, 'size_grp': 'mega',
            'ticker': f'T{p}', 'company_name': f'C{p}',
            'ret_exc_lead1m': rng.normal(0, 0.05),
        })
        rows.append(row)
    raw = pd.DataFrame(rows)

    chars_path = tmp_path / 'chars.parquet'
    raw.to_parquet(chars_path)
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'CHARS_PATH', chars_path)
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)

    panel = build_panel().set_index('permno')

    def expected_beta(betabab, beta60m, ivp):
        if not np.isnan(betabab):
            b1 = config.BETA_FP_W * np.clip(betabab, -1, 4) + config.BETA_FP_C
        elif not np.isnan(beta60m):
            b1 = 0.67 * beta60m + 0.33
        else:
            b1 = config.BETA_MISSING
        return config.BETA_INTERCEPT + config.BETA_SLOPE * b1 + config.BETA_IVOL * ivp

    ivp_by_p = {p: rank / 7 for rank, p in enumerate(range(7), start=1)}  # p0..p6 non-null, rank/7
    ivp_by_p[7] = 0.5  # p7: ivol_capm_252d missing -> fillna(0.5)

    for p, (betabab, beta60m, ivol252) in enumerate(specs):
        row = panel.loc[4000 + p]
        exp = expected_beta(betabab, beta60m, ivp_by_p[p])
        assert row['beta'] == pytest.approx(exp), f"p{p}: beta mismatch"

    assert panel.loc[4005, 'gics2'] == 'NA'
    assert panel.loc[4006, 'gics2'] == '16'


def test_beta_blume_formula_default(tmp_path, monkeypatch):
    """Pre-registered ('blume') beta model, A16's default (config.BETA_MODEL == 'blume'):
    beta = (1-BETA_SHRINK)*beta_60m + BETA_SHRINK*1.0, missing beta_60m -> 1.0 exactly. betabab_1260d
    and ivol_capm_252d (the A15 formula's inputs) are set to very different values per row here and
    must have no effect at all under this model."""
    monkeypatch.setattr(config, 'BETA_MODEL', 'blume')
    chars = config.load_char_list()
    rng = np.random.default_rng(11)
    eom = pd.Timestamp('2020-01-31')
    # (beta_60m, betabab_1260d, ivol_capm_252d) -- betabab/ivol vary a lot per row on purpose.
    specs = [
        (1.2, 1.0, 0.01),
        (0.5, 10.0, 0.02),
        (np.nan, -5.0, 0.03),  # beta_60m missing -> blume beta must be exactly 1.0
    ]
    rows = []
    for p, (beta60m, betabab, ivol252) in enumerate(specs):
        row = {c: rng.normal(0, 1) for c in chars}
        row.update({
            'permno': 6000 + p, 'eom': eom, 'date': eom,
            'prc': 20.0, 'me': 1_000_000.0,
            'gics': f'{10 + p}101010',
            'beta_60m': beta60m, 'betabab_1260d': betabab, 'ivol_capm_252d': ivol252,
            'dolvol_126d': 1000.0, 'size_grp': 'mega',
            'ticker': f'T{p}', 'company_name': f'C{p}',
            'ret_exc_lead1m': rng.normal(0, 0.05),
        })
        rows.append(row)
    raw = pd.DataFrame(rows)

    chars_path = tmp_path / 'chars.parquet'
    raw.to_parquet(chars_path)
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'CHARS_PATH', chars_path)
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)

    panel = build_panel().set_index('permno')

    for p, (beta60m, betabab, ivol252) in enumerate(specs):
        exp = 1.0 if np.isnan(beta60m) else (1 - config.BETA_SHRINK) * beta60m + config.BETA_SHRINK
        assert panel.loc[6000 + p, 'beta'] == pytest.approx(exp), f"p{p}: beta mismatch"


def test_panel_cache_filename_includes_beta_model(tmp_path, monkeypatch):
    """build_panel()'s on-disk cache filename is settings-specific (panel_<BETA_MODEL>.parquet, A16):
    switching config.BETA_MODEL must never silently reuse a panel cached under the other beta
    model -- each model gets, and rebuilds into, its own cache file, and the other model's cache
    file is left untouched (not overwritten, not read back)."""
    chars = config.load_char_list()
    rng = np.random.default_rng(13)
    eom = pd.Timestamp('2020-01-31')
    rows = []
    for p, beta60m in enumerate([1.2, 0.5, 1.8]):
        # id/aux columns set AFTER the chars fill (some chars share names with id/aux columns --
        # see the module-level NOTE in src/data.py._build): must not be clobbered by the random fill.
        row = {c: rng.normal(0, 1) for c in chars}
        row.update({
            'permno': 7000 + p, 'eom': eom, 'date': eom,
            'prc': 20.0, 'me': 1_000_000.0,
            'gics': f'{10 + p}101010',
            'beta_60m': beta60m, 'betabab_1260d': 0.3 * (p + 1), 'ivol_capm_252d': 0.01 * (p + 1),
            'dolvol_126d': 1000.0, 'size_grp': 'mega',
            'ticker': f'T{p}', 'company_name': f'C{p}',
            'ret_exc_lead1m': rng.normal(0, 0.05),
        })
        rows.append(row)
    raw = pd.DataFrame(rows)
    chars_path = tmp_path / 'chars.parquet'
    raw.to_parquet(chars_path)
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'CHARS_PATH', chars_path)
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)

    monkeypatch.setattr(config, 'BETA_MODEL', 'blume')
    panel_blume = build_panel()
    assert (cache_dir / 'panel_blume.parquet').exists()
    assert not (cache_dir / 'panel_a15.parquet').exists()
    assert not (cache_dir / 'panel.parquet').exists()

    monkeypatch.setattr(config, 'BETA_MODEL', 'a15')
    panel_a15 = build_panel()
    assert (cache_dir / 'panel_a15.parquet').exists()
    assert (cache_dir / 'panel_blume.parquet').exists()  # untouched, still there

    # beta actually differs between the two cached panels for at least one permno -- proves
    # build_panel() recomputed under the new model rather than silently reading back the stale
    # blume-model cache.
    merged = panel_blume.set_index('permno')[['beta']].join(
        panel_a15.set_index('permno')[['beta']], lsuffix='_blume', rsuffix='_a15')
    assert not np.allclose(merged['beta_blume'], merged['beta_a15'])


def test_ivp_ranks_within_eom_separately(tmp_path, monkeypatch):
    """ivp (the ivol_capm_252d percentile blended into 'beta', A15) is a within-eom rank
    (`.groupby(df['eom']).rank(pct=True)`), not a global rank -- two eoms whose ivol_capm_252d
    values sit on completely different scales must still produce the SAME ivp (and hence the
    same 'beta', since betabab_1260d/b1 is held constant here) at matching within-eom rank
    positions. A global rank would instead let eom2's uniformly-larger ivol scale dominate and
    push its ivp values above eom1's at every rank position."""
    monkeypatch.setattr(config, 'BETA_MODEL', 'a15')  # ivp only feeds 'beta' under the A15 model
    chars = config.load_char_list()
    rng = np.random.default_rng(7)
    eom1 = pd.Timestamp('2020-01-31')
    eom2 = pd.Timestamp('2020-02-29')
    # same 4 within-eom ivol ranks in both months, but on wildly different absolute scales
    # (eom1: ~0.01-0.04, eom2: ~100-400) -- same betabab_1260d (b1 constant) in both months, so
    # any beta difference at a matching rank position can only come from ivp.
    ivol_by_eom = {eom1: [0.01, 0.02, 0.03, 0.04], eom2: [100.0, 200.0, 300.0, 400.0]}
    betabab = 1.0

    rows = []
    permno = 5000
    for eom, ivols in ivol_by_eom.items():
        for rank_pos, ivol252 in enumerate(ivols):
            row = {c: rng.normal(0, 1) for c in chars}
            row.update({
                'permno': permno, 'eom': eom, 'date': eom,
                'prc': 20.0, 'me': 1_000_000.0,
                'gics': f'{10 + rank_pos}101010',
                'beta_60m': 1.2, 'betabab_1260d': betabab, 'ivol_capm_252d': ivol252,
                'dolvol_126d': 1000.0, 'size_grp': 'mega',
                'ticker': f'T{permno}', 'company_name': f'C{permno}',
                'ret_exc_lead1m': rng.normal(0, 0.05),
            })
            rows.append(row)
            permno += 1
    raw = pd.DataFrame(rows)

    chars_path = tmp_path / 'chars.parquet'
    raw.to_parquet(chars_path)
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'CHARS_PATH', chars_path)
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)

    panel = build_panel().set_index('permno')

    beta_eom1 = panel.loc[5000:5003, 'beta'].to_numpy()
    beta_eom2 = panel.loc[5004:5007, 'beta'].to_numpy()
    # matching rank positions (both strictly increasing ivol -> rank 1..4 of 4 in each eom)
    # must give identical beta despite the 10,000x difference in ivol scale between the eoms.
    assert beta_eom1 == pytest.approx(beta_eom2)
    # sanity: ivp actually varies within an eom (beta isn't just constant regardless of ivol).
    assert len(set(np.round(beta_eom1, 8))) == 4


def test_beta_var_zero_under_blume_and_a15(synthetic_env, monkeypatch):
    """src/beta.py: 'blume' and 'a15' reproduce the pre-existing formulas exactly, with
    beta_var == 0 (no uncertainty estimate for those models)."""
    monkeypatch.setattr(config, 'BETA_MODEL', 'blume')
    panel_blume = build_panel()
    assert 'beta_var' in panel_blume.columns
    assert (panel_blume['beta_var'] == 0.0).all()

    monkeypatch.setattr(config, 'BETA_MODEL', 'a15')
    panel_a15 = build_panel()
    assert (panel_a15['beta_var'] == 0.0).all()


def _synthetic_raw_beta_calib(n_permnos=12, seed=5):
    """A larger synthetic panel (2015-01..2018-12, all 147 chars + 'ret') sized so
    beta.fit_beta_params has enough calibration rows (2015-01..2017-12 formation months) to fit
    -- used to exercise the 'fusion'/'kalman' beta models end to end through build_panel."""
    rng = np.random.default_rng(seed)
    chars = config.load_char_list()
    eoms = pd.date_range('2015-01-31', periods=48, freq='ME')  # 2015-01..2018-12
    mkt_ret = pd.Series(rng.normal(0.01, 0.04, len(eoms)), index=eoms)
    permnos = np.arange(2000, 2000 + n_permnos)
    true_beta = {p: rng.uniform(0.4, 1.6) for p in permnos}
    rows = []
    for eom in eoms:
        for i, p in enumerate(permnos):
            tb = true_beta[p]
            row = {c: rng.normal(0, 1) for c in chars}
            is_young = i < 2  # first two permnos: always missing beta_60m, like real young stocks
            row.update({
                'permno': p, 'eom': eom, 'date': eom,
                # constant 'me' across permnos: universe_mask's ME_CUTOFF_PCTILE=0.20 filter must
                # not systematically drop the "young" (low-index) permnos this test relies on.
                'prc': 20.0, 'me': 1_000_000.0,
                'gics': f'{10 + i % 3}101010',
                'beta_60m': np.nan if is_young else tb + rng.normal(0, 0.15),
                'betabab_1260d': tb + rng.normal(0, 0.15),
                'betadown_252d': tb + rng.normal(0, 0.2),
                'beta_dimson_21d': tb + rng.normal(0, 0.3),
                'ivol_capm_252d': abs(rng.normal(0.02, 0.01)),
                'age': 12 + i * 3, 'at_be': 1.5 + 0.05 * i,
                'dolvol_126d': 1000.0, 'size_grp': 'mega',
                'ticker': f'T{p}', 'company_name': f'C{p}',
                'ret': tb * mkt_ret.loc[eom] + rng.normal(0, 0.04),
                'ret_exc_lead1m': rng.normal(0, 0.05),
            })
            rows.append(row)
    return pd.DataFrame(rows), mkt_ret


@pytest.fixture()
def fusion_kalman_env(tmp_path, monkeypatch):
    raw, mkt_ret = _synthetic_raw_beta_calib()
    chars_path = tmp_path / 'chars.parquet'
    raw.to_parquet(chars_path)
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'CHARS_PATH', chars_path)
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)
    monkeypatch.setattr(data_mod, 'market_state', lambda: pd.DataFrame({'mkt_ret': mkt_ret}))
    # pre-fit and cache beta_params.json (needs 'ret', which build_panel's own raw load doesn't
    # fetch) so build_panel -> _build -> compute_betas('fusion'/'kalman') hits the cache instead
    # of trying its own (network-free but real-file) load.
    from src import beta as beta_module
    beta_module.fit_beta_params(raw)
    return raw


@pytest.mark.parametrize('model', ['fusion', 'kalman'])
def test_build_panel_fusion_kalman_end_to_end(fusion_kalman_env, monkeypatch, model):
    monkeypatch.setattr(config, 'BETA_MODEL', model)
    panel = build_panel()
    # A16 audit fix #4: fusion/kalman cache filenames carry a hash of beta_params.json's content
    # (panel_<model>_<8hex>.parquet), not the bare panel_<model>.parquet blume/a15 use.
    matches = list(config.CACHE_DIR.glob(f'panel_{model}_*.parquet'))
    assert len(matches) == 1
    assert not (config.CACHE_DIR / f'panel_{model}.parquet').exists()
    assert 'beta_var' in panel.columns
    assert (panel['beta_var'] > 0).all()  # fusion/kalman always carry positive uncertainty
    assert panel['beta'].notna().all()
    # the young permnos (2000, 2001; always missing beta_60m) must not be flatly imputed to 1.0
    # the way 'blume' would -- that's the bug src/beta.py exists to fix.
    young = panel.loc[panel['permno'].isin([2000, 2001])]
    assert not np.allclose(young['beta'], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# slow tests on real data
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_panel_unique_and_ranks():
    panel = build_panel()
    assert not panel.duplicated(subset=['permno', 'eom']).any()

    feats = feature_columns(panel)
    chars = config.load_char_list()
    char_feats = [c for c in feats if c in chars]
    vals = panel[char_feats].to_numpy()
    assert np.nanmin(vals) >= -1 - 1e-9
    assert np.nanmax(vals) <= 1 + 1e-9

    for banned in ('ret_exc_lead1m', 'target_month', 'stock_exret'):
        assert banned not in feats


@pytest.mark.slow
def test_real_stock_exret_matches_next_month_raw():
    panel = build_panel()
    raw = pd.read_parquet(config.CHARS_PATH, columns=['permno', 'eom', 'ret_exc'])
    raw['eom'] = pd.to_datetime(raw['eom'])

    sample = panel[['permno', 'eom', 'target_month', 'stock_exret']].dropna(subset=['stock_exret'])
    sample = sample.sample(n=min(5000, len(sample)), random_state=0)

    raw_next = raw.rename(columns={'eom': 'target_month', 'ret_exc': 'ret_exc_next'})
    merged = sample.merge(raw_next, on=['permno', 'target_month'], how='left')
    found = merged.dropna(subset=['ret_exc_next'])
    match = np.isclose(found['stock_exret'], found['ret_exc_next'], atol=1e-9)
    match_rate = match.mean() if len(found) else float('nan')
    print(f'stock_exret vs raw next-month ret_exc match rate: {match_rate:.4f} (n={len(found)}/{len(sample)})')
    assert len(found) > 0
    assert match_rate > 0.999


@pytest.mark.slow
def test_real_load_market_coverage():
    market = load_market()
    window = market.loc['2020-12-31':'2026-08-31']
    expected_months = pd.date_range('2020-12-31', '2026-08-31', freq='ME')
    missing = expected_months.difference(window.index)
    assert len(missing) == 0, f'missing months: {missing.tolist()}'
    assert not window[['tb3ms', 'rf_m', 'sp500_ret', 'sp500_exret', 'rf_pipe']].isna().any().any()


@pytest.mark.slow
def test_real_sp500_total_return_sanity():
    """Sanity-check the cached external market data against well-known reference figures."""
    market = load_market()

    y2022 = market.loc['2022-01-31':'2022-12-31', 'sp500_ret']
    assert len(y2022) == 12
    compounded_2022 = (1 + y2022).prod() - 1
    assert compounded_2022 == pytest.approx(-0.181, abs=0.005)

    march2020 = pd.Timestamp('2020-03-31')
    if march2020 in market.index:
        assert market.loc[march2020, 'sp500_ret'] == pytest.approx(-0.124, abs=0.005)
    else:
        pytest.skip('2020 not present in the cached market data')

    assert market.loc[pd.Timestamp('2021-01-31'), 'tb3ms'] < 0.2
    jul2023 = market.loc[pd.Timestamp('2023-07-31'), 'tb3ms']
    assert 5.0 <= jul2023 <= 5.5


@pytest.mark.slow
def test_real_panel_truncation_invariance():
    """Truncation invariance on real data: rebuilding the panel (in memory, via _build, without
    touching the on-disk cache) from data truncated at T, with the label at T wiped out, must give
    the same month-T feature/aux columns as the full cached panel (labels never affect features)."""
    T = pd.Timestamp('2022-12-31')
    full_panel = build_panel()

    chars = config.load_char_list()
    id_cols = ['permno', 'eom', 'date', 'prc', 'me', 'gics', 'beta_60m',
               'dolvol_126d', 'size_grp', 'ticker', 'company_name', 'ret_exc_lead1m']
    needed = list(dict.fromkeys(id_cols + chars))
    raw_full = load_chars(columns=needed)
    raw_trunc = raw_full.loc[raw_full['eom'] <= T].copy()
    raw_trunc.loc[raw_trunc['eom'] == T, 'ret_exc_lead1m'] = np.nan

    trunc_panel = data_mod._build(raw_trunc)

    full_T = full_panel.loc[full_panel['eom'] == T].set_index('permno').sort_index()
    trunc_T = trunc_panel.loc[trunc_panel['eom'] == T].set_index('permno').sort_index()
    assert len(full_T) > 0
    common = full_T.index.intersection(trunc_T.index)
    assert len(common) == len(full_T)  # same universe membership at T

    feats = feature_columns(full_panel)
    aux_cols = ['gics2', 'beta', 'size_z', 'me', 'size_grp', 'ticker', 'company_name']
    compare_cols = [c for c in feats + aux_cols if c in full_T.columns and c in trunc_T.columns]

    a = full_T.loc[common, compare_cols]
    b = trunc_T.loc[common, compare_cols]
    numeric_cols = [c for c in compare_cols if pd.api.types.is_numeric_dtype(a[c])]
    other_cols = [c for c in compare_cols if c not in numeric_cols]
    assert np.allclose(a[numeric_cols].to_numpy(dtype=float), b[numeric_cols].to_numpy(dtype=float),
                        equal_nan=True)
    assert (a[other_cols].astype(str).values == b[other_cols].astype(str).values).all()


@pytest.mark.slow
def test_real_market_state_truncation_invariance(monkeypatch):
    """Same truncation-invariance check as above, for market_state: data <= T only must give the
    same month-T row as the full-data build."""
    T = pd.Timestamp('2022-12-31')
    full_state = market_state()

    real_load_chars = data_mod.load_chars

    def truncated_load_chars(columns=None):
        raw = real_load_chars(columns=columns)
        return raw.loc[raw['eom'] <= T].copy()

    monkeypatch.setattr(data_mod, 'load_chars', truncated_load_chars)
    trunc_state = market_state()

    pd.testing.assert_series_equal(full_state.loc[T], trunc_state.loc[T], check_names=False)
