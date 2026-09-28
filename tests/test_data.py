import numpy as np
import pandas as pd
import pytest

from src import config
from src import data as data_mod
from src.data import (
    _rank_within_eom, universe_mask, build_panel, feature_columns, market_state, load_market,
    load_chars,
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

    panel = build_panel()
    assert f'miss_{target_char}' not in panel.columns


def test_beta_shrink_and_gics2_na(tmp_path, monkeypatch):
    """beta shrink formula (missing beta_60m -> exactly 1.0; present -> shrink formula on RAW
    beta_60m) and gics2 'NA' fill for missing gics."""
    chars = config.load_char_list()
    rng = np.random.default_rng(4)
    eom = pd.Timestamp('2020-01-31')
    n_permnos = 10
    rows = []
    for p in range(n_permnos):
        row = {c: rng.normal(0, 1) for c in chars}
        # id/aux columns set AFTER the chars fill: some chars share names with id/aux columns
        # ('prc', 'dolvol_126d', 'beta_60m') and must not be clobbered by the random fill above.
        row.update({
            'permno': 4000 + p, 'eom': eom, 'date': eom,
            'prc': 20.0, 'me': 1000.0 * (p + 1),
            'gics': None if p == 5 else f'{10 + p}101010',
            'beta_60m': np.nan if p == 6 else 1.0 + 0.1 * p,
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

    panel = build_panel()
    p5 = panel.loc[panel['permno'] == 4005].iloc[0]  # gics missing, beta_60m = 1.5 present
    p6 = panel.loc[panel['permno'] == 4006].iloc[0]  # gics present, beta_60m missing

    assert p5['gics2'] == 'NA'
    expected_beta_p5 = (1 - config.BETA_SHRINK) * 1.5 + config.BETA_SHRINK * 1.0
    assert p5['beta'] == pytest.approx(expected_beta_p5)

    assert p6['beta'] == pytest.approx(1.0)
    assert p6['gics2'] == '16'


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
    assert not window[['tb3ms', 'rf_m', 'sp500_ret', 'sp500_exret']].isna().any().any()


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
