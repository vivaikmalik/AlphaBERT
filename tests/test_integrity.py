"""tests/test_integrity.py -- SPEC section 8 leakage/rules integrity suite (+ section 10
amendments). This is the standalone suite a judge could run to sanity-check the whole pipeline's
rules; it is deliberately somewhat self-contained (a little overlap with tests/test_data.py,
tests/test_text.py, tests/test_models.py is intentional so this single file gives full coverage
of section 8 on its own), but skips re-deriving checks those files already do byte-for-byte on
synthetic data (e.g. the rank formula, clean_text boilerplate stripping).

Marked @pytest.mark.slow throughout except the schedule/feature-hygiene checks that need no real
parquet reads (per SPEC section 0: only tests touching real data are marked slow).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src import config, data, text, models, portfolio, evaluate as ev

SEED = config.SEED
MINI_N_PERMNOS = 600
T_TRUNC = pd.Timestamp('2021-06-30')


# =============================================================================
# 1. Target alignment
# =============================================================================
@pytest.mark.slow
def test_target_alignment_real():
    """stock_exret(t) == raw ret_exc(t+1) (sampled); target_month == eom + 1 month-end."""
    panel = data.build_panel()

    # target_month == eom + 1 month-end, on every row (cheap, not just sampled)
    assert (panel['target_month'] == panel['eom'] + pd.offsets.MonthEnd(1)).all()

    raw = pd.read_parquet(config.CHARS_PATH, columns=['permno', 'eom', 'ret_exc'])
    raw['eom'] = pd.to_datetime(raw['eom'])

    sample = panel[['permno', 'eom', 'target_month', 'stock_exret']].dropna(subset=['stock_exret'])
    sample = sample.sample(n=min(5000, len(sample)), random_state=SEED)

    raw_next = raw.rename(columns={'eom': 'target_month', 'ret_exc': 'ret_exc_next'})
    merged = sample.merge(raw_next, on=['permno', 'target_month'], how='left')
    found = merged.dropna(subset=['ret_exc_next'])
    match = np.isclose(found['stock_exret'], found['ret_exc_next'], atol=1e-9)
    match_rate = match.mean() if len(found) else float('nan')
    print(f'[integrity] stock_exret vs raw ret_exc(t+1) match rate: {match_rate:.4f} (n={len(found)})')
    assert len(found) > 0
    assert match_rate > 0.999


# =============================================================================
# 2. Truncation invariance (panel / market_state / text), all at T = 2021-06-30
# =============================================================================
@pytest.mark.slow
def test_truncation_invariance_panel_real():
    full_panel = data.build_panel()

    chars = config.load_char_list()
    id_cols = ['permno', 'eom', 'date', 'prc', 'me', 'gics', 'beta_60m',
               'dolvol_126d', 'size_grp', 'ticker', 'company_name', 'ret_exc_lead1m']
    needed = list(dict.fromkeys(id_cols + chars))
    raw_full = data.load_chars(columns=needed)
    raw_trunc = raw_full.loc[raw_full['eom'] <= T_TRUNC].copy()
    raw_trunc.loc[raw_trunc['eom'] == T_TRUNC, 'ret_exc_lead1m'] = np.nan

    trunc_panel = data._build(raw_trunc)

    full_T = full_panel.loc[full_panel['eom'] == T_TRUNC].set_index('permno').sort_index()
    trunc_T = trunc_panel.loc[trunc_panel['eom'] == T_TRUNC].set_index('permno').sort_index()
    assert len(full_T) > 0
    common = full_T.index.intersection(trunc_T.index)
    assert len(common) == len(full_T)  # same universe membership at T

    feats = data.feature_columns(full_panel)
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
def test_truncation_invariance_market_state_real(monkeypatch):
    full_state = data.market_state()

    real_load_chars = data.load_chars

    def truncated_load_chars(columns=None):
        raw = real_load_chars(columns=columns)
        return raw.loc[raw['eom'] <= T_TRUNC].copy()

    monkeypatch.setattr(data, 'load_chars', truncated_load_chars)
    trunc_state = data.market_state()

    pd.testing.assert_series_equal(full_state.loc[T_TRUNC], trunc_state.loc[T_TRUNC], check_names=False)


@pytest.mark.slow
def test_truncation_invariance_text_real(tmp_path, monkeypatch):
    """Not covered elsewhere on real data: tests/test_text.py's truncation-invariance test uses a
    tiny synthetic filings frame. Here we truncate the real 8-K archive at filing_date <= T and
    check month-T text features match the full build."""
    full_feat = text.build_text_features()

    filings = pq.read_table(
        config.FILINGS_PATH, columns=['document_id', 'permno', 'filing_date', 'items']
    ).to_pandas()
    filings['filing_date'] = pd.to_datetime(filings['filing_date'])
    trunc_filings = filings.loc[filings['filing_date'] <= T_TRUNC].copy()
    trunc_path = tmp_path / 'filings_trunc.parquet'
    trunc_filings.to_parquet(trunc_path, index=False)
    monkeypatch.setattr(config, 'FILINGS_PATH', trunc_path)

    trunc_feat = text.build_text_features()

    full_T = full_feat[full_feat['eom'] == T_TRUNC].set_index('permno').sort_index()
    trunc_T = trunc_feat[trunc_feat['eom'] == T_TRUNC].set_index('permno').sort_index()
    assert len(full_T) > 0
    common = full_T.index.intersection(trunc_T.index)
    assert len(common) == len(full_T)

    cols = [c for c in full_T.columns if c in trunc_T.columns]
    pd.testing.assert_frame_equal(full_T.loc[common, cols], trunc_T.loc[common, cols], check_dtype=False)


# =============================================================================
# 3. Feature hygiene
# =============================================================================
@pytest.mark.slow
def test_feature_hygiene_real():
    panel = data.build_panel()
    feats = set(data.feature_columns(panel)) | set(text.TEXT_FEATURES)

    banned = {'stock_exret', 'ret_exc_lead1m', 'target_month', 'has_filing', 'ticker', 'company_name'}
    assert banned.isdisjoint(feats), f'banned columns leaked into features: {banned & feats}'
    assert not any(f.endswith('_raw') for f in feats), \
        f'*_raw aux columns leaked into features: {[f for f in feats if f.endswith("_raw")]}'

    aux_or_id = {'gics2', 'beta', 'size_z', 'me', 'size_grp', 'date', 'eom', 'permno'}
    assert aux_or_id.isdisjoint(feats)


# =============================================================================
# 4. Schedule (per year + inner holdout inside train)
# =============================================================================
def test_schedule_and_inner_holdout():
    """Reproduces tests/test_models.py::test_splits_schedule_matches_spec's per-year/train<valid<test
    and test-month-union checks, plus the inner-holdout-inside-train property (A2/A13: the last 12
    target months of train used by models._fit_specialist) which that test doesn't cover."""
    eom = pd.date_range('2015-01-31', '2026-07-31', freq='ME')
    permnos = np.arange(5)
    df = pd.DataFrame({
        'permno': np.tile(permnos, len(eom)),
        'eom': np.repeat(eom, len(permnos)),
    })
    df['target_month'] = df['eom'] + pd.offsets.MonthEnd(1)
    df['stock_exret'] = np.random.RandomState(0).normal(size=len(df))

    test_month_union = set()
    for y in config.TEST_YEARS:
        train_mask, valid_mask, test_mask = models.splits(df, y)
        train_tm = df.loc[train_mask, 'target_month']
        valid_tm = df.loc[valid_mask, 'target_month']
        test_tm = df.loc[test_mask, 'target_month']
        assert train_tm.max() < valid_tm.min() < test_tm.min()
        test_month_union |= set(test_tm.unique())

        # inner holdout (last 12 target months of train) sits strictly inside train, at its tail
        months = np.sort(train_tm.unique())
        ho = min(12, max(1, len(months) - 1))
        holdout_months = months[-ho:]
        inner_train_months = months[:-ho]
        assert set(holdout_months) <= set(months)
        assert inner_train_months.max() < holdout_months.min()
        assert holdout_months.max() == months.max()

    assert test_month_union == set(pd.date_range('2021-01-31', '2026-08-31', freq='ME'))


# =============================================================================
# Shared fixture for the mini end-to-end / shuffled-label checks
# =============================================================================
@pytest.fixture(scope='module')
def mini_panel():
    """Real panel + text features restricted to a random 600-permno subset (seeded), plus the
    real (full-universe) market state. Shared across the mini-pipeline and shuffled-label tests
    to avoid rebuilding it twice."""
    rng = np.random.default_rng(SEED)
    full = data.build_panel()
    uniq_permnos = full['permno'].unique()
    keep = rng.choice(uniq_permnos, size=min(MINI_N_PERMNOS, len(uniq_permnos)), replace=False)
    panel = full[full['permno'].isin(keep)].reset_index(drop=True)
    panel = text.add_text_features(panel)
    state = data.market_state()
    return panel, state


def _patch_mini_config(monkeypatch):
    monkeypatch.setattr(config, 'TEST_YEARS', [2021])
    monkeypatch.setattr(config, 'TEST_END', pd.Timestamp('2021-12-31'))
    monkeypatch.setattr(config, 'N_JOBS', 4)
    # Speed knobs only (never touched by src/ for correctness): the production ROUND_GRID (up to
    # 1000 rounds x 2 leaf configs x 7 model fits) and the 15-candidate Lasso/ElasticNet alpha
    # grids take ~20-25 min per run_all() call even on this 600-permno subset -- far outside this
    # suite's per-test budget. This is a mechanics/interface smoke test (see module docstring), not
    # a performance benchmark, so a much smaller grid is fine here.
    monkeypatch.setattr(models, 'ROUND_GRID', (30, 60))
    monkeypatch.setattr(models, 'RIDGE_ALPHAS', np.logspace(-2, 4, 5))
    monkeypatch.setattr(models, 'LASSO_ALPHAS', np.logspace(-6, -2, 3))
    monkeypatch.setattr(models, 'ENET_ALPHAS', np.logspace(-6, -2, 3))
    monkeypatch.setattr(models, 'MAX_ROWS_PENALIZED', 15_000)
    # models.fit_baselines() actually iterates models._PENALIZED_SPECS, a list built ONCE at
    # src/models.py import time from the (name, class, alpha_grid, kwargs, max_rows) tuples --
    # it captures the *objects* RIDGE_ALPHAS/LASSO_ALPHAS/ENET_ALPHAS/MAX_ROWS_PENALIZED held at
    # that time. Patching those four module attributes above (as this helper did before) rebinds
    # only the *names*; _PENALIZED_SPECS keeps its original references, so fit_baselines silently
    # keeps running the full production 15-alpha Lasso/ElasticNet grids, max_iter=5000, on the
    # *entire* (non-subsampled) training window every time (MAX_ROWS_PENALIZED=100_000 baked in,
    # far above this fixture's ~25-30k train rows) -- the actual source of the "~20-25 min" cost
    # noted above, not the LightGBM grid. Rebuild _PENALIZED_SPECS from the patched values (and cut
    # max_iter, since the final refit is never subsampled) so the speed knobs take effect.
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    monkeypatch.setattr(models, '_PENALIZED_SPECS', [
        ('ridge', Ridge, models.RIDGE_ALPHAS, {}, None),
        ('lasso', Lasso, models.LASSO_ALPHAS, {'max_iter': 200}, models.MAX_ROWS_PENALIZED),
        ('enet', ElasticNet, models.ENET_ALPHAS, {'max_iter': 200, 'l1_ratio': 0.5},
         models.MAX_ROWS_PENALIZED),
    ])


# =============================================================================
# 5. Mini end-to-end pipeline
# =============================================================================
@pytest.mark.slow
def test_mini_end_to_end_pipeline(mini_panel, monkeypatch, tmp_path):
    """Runs the full MAIN.py-shaped pipeline (models.run_all -> smooth -> backtest -> attach_labels
    -> write_submission -> evaluate.run_evaluation -> calibrate -> missing_return_sensitivity) on a
    600-permno subset for test_year=2021 only, to prove the interfaces compose without the
    50-minute full run. Mechanics only -- no test-period performance is reported here.

    Previously xfail (A14 pending): portfolio.optimize_month could be infeasible even after the
    relaxation ladder when top/bottom-N_CAND candidates were sector-lopsided. That is fixed
    (sector-demeaned candidate selection, docs/SPEC.md A14) and this test now asserts real passes,
    not just tolerates failure."""
    import time
    t0 = time.time()

    panel, state = mini_panel
    _patch_mini_config(monkeypatch)
    monkeypatch.setattr(config, 'N_CAND', 150)
    # SHORT_SCREEN off: on this 600-permno subset, the me/dolvol tradability screen leaves the
    # short leg sector-lopsided enough to break sector neutrality (this fixture is a random
    # 600-name subset, far smaller/sparser than the real universe SHORT_SCREEN was tuned
    # against). This is a mechanics/interface smoke test (see module docstring), not a
    # SHORT_SCREEN check -- that's covered on its own synthetic fixture in
    # tests/test_portfolio.py::test_short_screen_excludes_illiquid_names_from_short_leg_only.
    monkeypatch.setattr(config, 'SHORT_SCREEN', False)

    preds = models.run_all(panel, state, cache_dir=tmp_path / 'cache')
    print(f'[integrity] mini run_all done in {time.time() - t0:.1f}s')

    test_preds = preds[preds['split'] == 'test']
    formation_months = sorted(test_preds['eom'].unique())
    assert formation_months[0] == pd.Timestamp('2020-12-31')
    assert formation_months[-1] == pd.Timestamp('2021-11-30')

    signal_df = test_preds[['permno', 'eom']].copy()
    signal_df['signal'] = portfolio.smooth(test_preds, 'pred_ew')

    market = data.load_market()
    holdings, returns = portfolio.backtest(signal_df, panel, market)

    # ---- holdings mechanics -------------------------------------------------
    assert (holdings['month'] == holdings['eom'] + pd.offsets.MonthEnd(1)).all()
    expected_holding_months = {m + pd.offsets.MonthEnd(1) for m in formation_months}
    assert set(holdings['month'].unique()) == expected_holding_months

    for mth, g in holdings.groupby('month'):
        nz = g.loc[g['weight'] != 0]
        n_names = len(nz)
        assert 100 <= n_names <= 500, f'{mth}: n_names={n_names}'
        long_w = nz.loc[nz['weight'] > 0, 'weight']
        short_w = nz.loc[nz['weight'] < 0, 'weight']
        # NET_MODE='beta' (default): legs sum to 1+n/2 / -(1-n/2) for the month's own solved
        # net exposure n, rather than the fixed +-1 of NET_MODE='dollar'.
        n_val = float(long_w.sum() + short_w.sum())
        assert abs(n_val) <= config.NET_CAP + 1e-4, f'{mth}: |net| exceeds NET_CAP'
        assert long_w.sum() == pytest.approx(1.0 + n_val / 2.0, abs=1e-4), f'{mth}: long leg off target'
        assert short_w.sum() == pytest.approx(-(1.0 - n_val / 2.0), abs=1e-4), f'{mth}: short leg off target'
        assert nz['weight'].abs().max() <= config.MAX_WEIGHT + 1e-6

    # ---- returns mechanics ---------------------------------------------------
    expected_cols = {
        'long_ret', 'short_ret', 'ls_ret', 'rf_m', 'total_ret', 'bench_ret', 'active_ret',
        'sp500_ret', 'sp500_exret', 'n_long', 'n_short', 'gross', 'net', 'beta_exante',
        'turnover', 'cost', 'total_ret_net', 'active_ret_net', 'missing_ret_weight',
        'filer_net', 'first_month', 'relax', 'net_target',
    }
    assert expected_cols <= set(returns.columns), f'missing columns: {expected_cols - set(returns.columns)}'
    assert not returns['total_ret'].isna().any()

    # filer-net-neutral (A10): within SECTOR_TOL of n*filer_share (relative constraint under
    # NET_MODE='beta', see src/portfolio.py optimize_month/_group_shares; conservatively
    # bounded here by NET_CAP since s_g <= 1), widened x2 on rows where relax widened 'sector'
    allowed = np.where(returns['relax'].str.contains('sector', na=False), 2 * config.SECTOR_TOL,
                        config.SECTOR_TOL) + returns['net_target'].abs().to_numpy() + 1e-5
    assert (returns['filer_net'].abs() <= allowed).all()

    # ---- write_submission ------------------------------------------------------
    label_panel, filing_labels = portfolio.load_label_sources()
    holdings = portfolio.attach_labels(holdings, label_panel, filing_labels)

    sub_dir = tmp_path / 'submission'
    sub_dir.mkdir()
    monkeypatch.setattr(config, 'SUB_DIR', sub_dir)
    portfolio.write_submission(holdings, returns)

    h = pd.read_csv(sub_dir / 'holdings.csv')
    assert (pd.to_datetime(h['Date']).dt.day == 1).all()
    for d, g in h.groupby('Date')['WEIGHT']:
        # NET_MODE='beta' (default): legs sum to 100*(1+n/2) / -100*(1-n/2), n derived from
        # the month's own weights (see portfolio._round_submission_weights).
        n = float(g.sum()) / 100.0
        assert abs(n) <= 0.50 + 1e-6, f'{d}: net exposure exceeds the +-50% mandate'
        assert abs(g[g > 0].sum() - 100.0 * (1.0 + n / 2.0)) < 1e-6, f'{d}: long WEIGHT off target'
        assert abs(g[g < 0].sum() + 100.0 * (1.0 - n / 2.0)) < 1e-6, f'{d}: short WEIGHT off target'
    assert h['WEIGHT'].abs().max() <= config.MAX_WEIGHT * 100.0 + 1e-6

    # ---- evaluate: proves the interface composes (TEST_END patched to 2021-12-31 so
    # evaluate's own all-68-months assert matches this mini run's 12-month window) --------
    table_dir, fig_dir = tmp_path / 'tables', tmp_path / 'figures'
    table_dir.mkdir()
    fig_dir.mkdir()
    monkeypatch.setattr(config, 'TABLE_DIR', table_dir)
    monkeypatch.setattr(config, 'FIG_DIR', fig_dir)
    gate_coefs = pd.read_parquet(tmp_path / 'cache' / 'gate_coefs.parquet')
    headline = ev.run_evaluation(returns, holdings, panel, gate_coefs=gate_coefs, state=state)
    assert {'ir', 'sharpe', 'cagr', 'max_dd', 'alpha_ann', 'beta'} <= set(headline)

    # ---- calibrate + missing_return_sensitivity: exercised on a few VALIDATION months (the
    # mini run_all() call above saved valid-split rows too, since _patch_mini_config sets
    # TEST_YEARS=[2021] and run_all only saves 'valid' rows for TEST_YEARS[0], A11) -- a handful
    # of months is enough to prove the interface (calibrate's 4x4 penalty grid re-runs backtest()
    # per cell, so this stays cheap) ---------------------------------------------------------
    valid_preds = preds.loc[preds['split'] == 'valid', ['permno', 'eom', 'pred_ew']].copy()
    assert len(valid_preds), 'mini run_all produced no valid-split rows to calibrate on'
    few_months = sorted(valid_preds['eom'].unique())[:3]
    valid_preds = valid_preds[valid_preds['eom'].isin(few_months)]
    valid_signal = valid_preds[['permno', 'eom']].copy()
    valid_signal['signal'] = portfolio.smooth(valid_preds, 'pred_ew')

    l2, tc, calib_table = portfolio.calibrate(valid_signal, panel, market)
    assert {'l2', 'tc', 'avg_names_per_side', 'avg_turnover', 'score'} <= set(calib_table.columns)
    assert len(calib_table) == 16  # 4 L2_PENALTY values x 4 TURNOVER_PENALTY values
    assert l2 > 0 and tc > 0
    assert (calib_table['score'] >= 0).all()

    sensitivity = portfolio.missing_return_sensitivity(holdings, panel, returns)
    assert {'total_ret', 'active_ret', 'ls_ret', 'long_ret', 'short_ret'} <= set(sensitivity.columns)
    assert sensitivity.index.equals(returns.index)
    assert not sensitivity['total_ret'].isna().any()
    # columns missing_return_sensitivity's own docstring says carry over unchanged
    for col in ('turnover', 'cost', 'gross', 'net'):
        pd.testing.assert_series_equal(sensitivity[col], returns[col])

    print(f'[integrity] mini end-to-end pipeline total time: {time.time() - t0:.1f}s')


# =============================================================================
# 6. Shuffled-label test
# =============================================================================
@pytest.mark.slow
def test_shuffled_label_ic_real(mini_panel, monkeypatch, tmp_path):
    """R=6 INDEPENDENT shuffled-label fits, not one. Rationale (why a fixed +-0.02 bound on a
    single fit is statistically wrong): a single shuffled-label fit is a random function of the
    real (unshuffled) features -- shuffling stock_exret within month for train+valid destroys the
    true label relationship, but gradient boosting can still overfit its inner-holdout noise onto
    one specialist and, by pure chance, land a modest (~0.02-0.08) spurious IC against the REAL
    test-year labels of either sign (test_models.py::test_shuffled_label_ic_small needed 16
    independent reps of one cheap specialist fit for a tight near-zero null check for the same
    reason). A prior draft of this test observed exactly that: two single-rep runs of the full
    6-specialist run_all() gave -0.0229 and +0.0383 -- inconsistent sign, both plausible noise, but
    neither individually distinguishable from a small one-off leak at a fixed +-0.02 threshold.

    Fix: treat R independent reps' mean-test-IC-of-pred_ew as an i.i.d. sample of a mean-zero null
    (a within-month-shuffled label carries no genuine information, so nothing should survive to
    test-year IC in expectation) and test the SAMPLE MEAN against a bound scaled by the observed
    between-rep spread: |mean over reps| < 2.5 * sd_over_reps / sqrt(R) + 0.005. This is a
    (loose, two-sided ~99%) t-style bound on the mean; the +0.005 floor keeps the bound from
    collapsing to ~0 if sd happens to be tiny by chance. Each rep uses an independent seed and the
    shared _patch_mini_config's fast LightGBM/baseline settings (~25-30s/rep on this 600-permno
    subset with those settings actually wired up -- see _patch_mini_config's own comment on the
    _PENALIZED_SPECS fix), so R=6 full run_all() calls comfortably fit this suite's runtime
    budget."""
    import time
    t0 = time.time()

    panel, state = mini_panel
    _patch_mini_config(monkeypatch)

    train_mask, valid_mask, _ = models.splits(panel, 2021)
    shuffle_mask = (train_mask | valid_mask) & panel['stock_exret'].notna()
    idx = panel.index[shuffle_mask.values]

    R = 6
    rep_ics = []
    for rep in range(R):
        panel_rep = panel.copy()
        rng = np.random.default_rng(SEED + 1000 * (rep + 1))  # independent seed per rep
        shuffled_vals = panel_rep.loc[idx].groupby('eom')['stock_exret'].transform(
            lambda s: rng.permutation(s.values))
        panel_rep.loc[idx, 'stock_exret'] = shuffled_vals.values

        preds = models.run_all(panel_rep, state, cache_dir=tmp_path / f'cache_shuffled_{rep}')
        test_df = preds[preds['split'] == 'test']
        ic = float(models.monthly_ic(test_df, 'pred_ew').mean())
        rep_ics.append(ic)
        print(f'[integrity] shuffled-label rep {rep}/{R} mean test IC (pred_ew): {ic:.4f} '
              f'(elapsed {time.time() - t0:.1f}s)')

    rep_ics = np.array(rep_ics)
    mean_ic = float(rep_ics.mean())
    sd_ic = float(rep_ics.std(ddof=1))
    bound = 2.5 * sd_ic / np.sqrt(R) + 0.005
    print(f'[integrity] shuffled-label per-rep IC: {[round(x, 4) for x in rep_ics]}; '
          f'mean={mean_ic:.4f} sd={sd_ic:.4f} bound=+-{bound:.4f} '
          f'(total {time.time() - t0:.1f}s)')
    assert abs(mean_ic) < bound, (
        f'shuffled-label mean IC over {R} independent reps {mean_ic:.4f} exceeds the '
        f'noise-scaled bound +-{bound:.4f} (per-rep sd={sd_ic:.4f})')


# =============================================================================
# 7. Label hygiene
# =============================================================================
def _independent_asof(rows, source, date_col):
    """Re-derive the backward-asof match ourselves (plain pd.merge_asof, not attach_labels' own
    _asof_fill helper) and keep the matched source date -- so we can check the date directly,
    rather than by (permno, ticker, company_name) identity. Identity matching is unsound here:
    ticker/company_name are stable for years, so a naive identity-join against label_panel/
    filing_labels pulls in every date that permno ever had that name, most of them after `eom`,
    which is not evidence attach_labels used a future date -- it's just the name not changing."""
    src = source.dropna(subset=['ticker', 'company_name']).sort_values(date_col)
    src = src.rename(columns={date_col: 'src_date'})[['permno', 'src_date', 'ticker', 'company_name']]
    left = rows[['permno', 'eom']].copy()
    left['_order'] = np.arange(len(left))
    left = left.sort_values('eom')
    merged = pd.merge_asof(left, src, left_on='eom', right_on='src_date', by='permno', direction='backward')
    return merged.sort_values('_order').reset_index(drop=True)


@pytest.mark.slow
def test_attach_labels_never_uses_future_label_real():
    """attach_labels never uses a label dated after eom, on a real-data sample: independently
    recompute the backward-asof match (see _independent_asof) and confirm (a) it agrees with
    attach_labels' actual output -- proving attach_labels is correctly wired to a backward-asof
    join at real-data scale -- and (b) the matched date is never after eom."""
    panel = data.build_panel()
    label_panel, filing_labels = portfolio.load_label_sources()

    sample = panel[['permno', 'eom']].dropna().sample(n=500, random_state=SEED)
    holdings = pd.DataFrame({
        'month': sample['eom'] + pd.offsets.MonthEnd(1),
        'eom': sample['eom'].values,
        'permno': sample['permno'].values,
        'weight': 0.01,
        'ticker': None, 'company_name': None, 'label_source': None,
    })

    out = portfolio.attach_labels(holdings, label_panel, filing_labels).reset_index(drop=True)

    raw_rows = out[out['label_source'] == 'raw_panel']
    if len(raw_rows):
        recomputed = _independent_asof(raw_rows, label_panel, 'eom')
        assert (recomputed['ticker'].values == raw_rows['ticker'].values).all()
        assert (recomputed['company_name'].values == raw_rows['company_name'].values).all()
        assert (recomputed['src_date'] <= recomputed['eom']).all(), \
            'attach_labels used a raw_panel label dated after eom'

    filing_rows = out[out['label_source'] == 'filing']
    if len(filing_rows):
        recomputed = _independent_asof(filing_rows, filing_labels, 'filing_date')
        assert (recomputed['ticker'].values == filing_rows['ticker'].values).all()
        assert (recomputed['company_name'].values == filing_rows['company_name'].values).all()
        assert (recomputed['src_date'] <= recomputed['eom']).all(), \
            'attach_labels used a filing label dated after eom'

    # regression probe: an injected future-only label for a real permno/eom must never surface
    probe = sample.iloc[0]
    future_row = pd.DataFrame({
        'permno': [probe['permno']], 'eom': [probe['eom'] + pd.offsets.MonthEnd(1)],
        'ticker': ['FUTURE_SENTINEL'], 'company_name': ['Future Sentinel Co'],
    })
    label_panel2 = pd.concat([label_panel, future_row], ignore_index=True)
    probe_holdings = pd.DataFrame({
        'month': [probe['eom'] + pd.offsets.MonthEnd(1)], 'eom': [probe['eom']],
        'permno': [probe['permno']], 'weight': [0.01],
        'ticker': [None], 'company_name': [None], 'label_source': [None],
    })
    probe_out = portfolio.attach_labels(probe_holdings, label_panel2, filing_labels)
    assert probe_out['ticker'].iloc[0] != 'FUTURE_SENTINEL'


# =============================================================================
# 8. Universe invariance to NaN-ing ret_exc_lead1m
# =============================================================================
@pytest.mark.slow
def test_universe_invariance_to_nan_ret_exc_lead1m_real():
    raw = data.load_chars(columns=['permno', 'eom', 'prc', 'me', 'ret_exc_lead1m'])
    mask1 = data.universe_mask(raw)

    raw2 = raw.copy()
    rng = np.random.default_rng(SEED)
    nan_idx = rng.choice(len(raw2), size=len(raw2) // 3, replace=False)
    raw2.loc[raw2.index[nan_idx], 'ret_exc_lead1m'] = np.nan
    mask2 = data.universe_mask(raw2)

    assert (mask1 == mask2).all()
