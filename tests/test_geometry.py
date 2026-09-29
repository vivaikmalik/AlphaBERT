import numpy as np
import pandas as pd
import pytest

from src import config, geometry, text


def _synthetic(seed, n, D=8, n_permno=5, n_blobs=3,
               date_start='2018-01-01', date_end='2021-12-31'):
    """Synthetic FinBERT-like embeddings: n_blobs well-separated Gaussian blobs in D dims."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(n_blobs, D))
    blob_idx = rng.integers(0, n_blobs, size=n)
    emb = (centers[blob_idx] + rng.normal(scale=0.4, size=(n, D))).astype(np.float32)
    permno = rng.integers(1, n_permno + 1, size=n)
    days = pd.date_range(date_start, date_end, freq='D')
    filing_date = pd.to_datetime(rng.choice(days, size=n))
    ids_df = pd.DataFrame({
        'document_id': [f'd{i}' for i in range(n)],
        'permno': permno,
        'filing_date': filing_date,
    })
    return ids_df, emb


def _set_geom_config(monkeypatch, K=3, pca_dims=5, fit_end='2020-06-30', lookback=6, seed=0):
    monkeypatch.setattr(config, 'GEOM_K', K)
    monkeypatch.setattr(config, 'GEOM_PCA_DIMS', pca_dims)
    monkeypatch.setattr(config, 'GEOM_FIT_END', pd.Timestamp(fit_end))
    monkeypatch.setattr(config, 'GEOM_NOVELTY_LOOKBACK_MONTHS', lookback)
    monkeypatch.setattr(config, 'SEED', seed)


def _patch_existing_embeddings(monkeypatch, tmp_path, ids_df, emb):
    """Make build_geometry_features() think the embedding cache exists and load our synthetic
    data instead of the real files."""
    emb_path = tmp_path / 'finbert_emb_L512.npy'
    ids_path = tmp_path / 'finbert_emb_L512_ids.parquet'
    emb_path.touch()
    ids_path.touch()
    monkeypatch.setattr(geometry, '_emb_paths', lambda: (emb_path, ids_path))
    monkeypatch.setattr(text, 'load_embeddings', lambda: (ids_df, emb), raising=False)


# ---------------------------------------------------------------- 1. fit only on <= GEOM_FIT_END
def test_fit_ignores_post_cutoff_embeddings(monkeypatch):
    _set_geom_config(monkeypatch, K=3, pca_dims=5, fit_end='2020-06-30', seed=0)
    ids_df, emb = _synthetic(seed=1, n=180, D=8, n_blobs=3)

    pca1, kmeans1, X1 = geometry._fit(ids_df, emb)

    # perturb only the post-cutoff rows with unrelated random data
    rng = np.random.default_rng(99)
    emb2 = emb.copy()
    post_mask = (ids_df['filing_date'] > config.GEOM_FIT_END).to_numpy()
    emb2[post_mask] = rng.normal(scale=50.0, size=(post_mask.sum(), emb.shape[1])).astype(np.float32)

    pca2, kmeans2, X2 = geometry._fit(ids_df, emb2)

    assert np.allclose(pca1.components_, pca2.components_)
    assert np.allclose(kmeans1.cluster_centers_, kmeans2.cluster_centers_)
    # fit-set (pre-cutoff) transformed vectors are also unaffected
    pre_mask = ~post_mask
    assert np.allclose(X1[pre_mask], X2[pre_mask])


# ---------------------------------------------------------------- 2. novelty
def test_novelty_strictly_earlier_within_lookback(monkeypatch):
    _set_geom_config(monkeypatch, K=2, lookback=6)

    # permno 1, sorted by date: A(Jan31)=[1,0], B(Mar31)=[0,1], D(Sep15)=[.6,.8], C(Oct31)=[1,0].
    # lookback=6mo. B's cutoff=2019-09-30 -> only A eligible. D's cutoff=2020-03-15 -> only B
    # eligible (A is 1 day too early). C's cutoff=2020-04-30 -> only D eligible (A, B too early).
    # Row order deliberately NOT sorted by date, to check internal sorting.
    ids_df = pd.DataFrame({
        'document_id': ['D', 'A', 'C', 'B', 'E'],
        'permno': [1, 1, 1, 1, 2],
        'filing_date': pd.to_datetime([
            '2020-09-15', '2020-01-31', '2020-10-31', '2020-03-31', '2020-05-31',
        ]),
    })
    X = np.array([
        [0.6, 0.8],   # D
        [1.0, 0.0],   # A
        [1.0, 0.0],   # C
        [0.0, 1.0],   # B
        [1.0, 0.0],   # E (permno 2, only filing -> no prior)
    ])

    novelty = geometry._novelty(ids_df, X)
    nov = dict(zip(ids_df['document_id'], novelty))

    assert np.isnan(nov['A'])          # firm's first filing: nothing earlier
    assert np.isclose(nov['B'], 1.0)   # only A eligible, cos(B,A)=0 -> novelty=1
    assert np.isclose(nov['D'], 0.2)   # only B eligible, cos(D,B)=0.8 -> novelty=0.2
    assert np.isclose(nov['C'], 0.4)   # only D eligible (A, B out of lookback), cos(C,D)=0.6
    assert np.isnan(nov['E'])          # no other filing for this firm at all


# ---------------------------------------------------------------- 3. wave: past months only
def _wave_inputs(month_cluster_counts, K):
    rows = []
    for month, counts in month_cluster_counts.items():
        for cluster, n in counts.items():
            rows.extend([(pd.Timestamp(month) + pd.offsets.MonthEnd(0), cluster)] * n)
    df = pd.DataFrame(rows, columns=['eom', 'cluster'])
    return pd.DataFrame({'eom': df['eom']}), df['cluster'].to_numpy()


def test_wave_uses_past_months_only(monkeypatch):
    monkeypatch.setattr(config, 'GEOM_K', 2)

    base_counts = {
        '2021-01-15': {0: 3, 1: 1},
        '2021-02-15': {0: 1, 1: 0},
        '2021-03-15': {0: 5, 1: 2},
    }
    ids_df, labels = _wave_inputs(base_counts, K=2)
    wave = geometry._wave_z(ids_df, labels)

    df = pd.DataFrame({'eom': ids_df['eom'], 'cluster': labels, 'wave': wave})
    # January: no past months -> base = 0
    jan_c0 = df[(df['eom'] == pd.Timestamp('2021-01-31')) & (df['cluster'] == 0)]['wave'].iloc[0]
    assert np.isclose(jan_c0, (3 - 0) / np.sqrt(0 + 1))
    # March cluster 0: base = mean(Jan=3, Feb=1) = 2
    mar_c0 = df[(df['eom'] == pd.Timestamp('2021-03-31')) & (df['cluster'] == 0)]['wave'].iloc[0]
    assert np.isclose(mar_c0, (5 - 2) / np.sqrt(2 + 1))

    # Add a later month; earlier months' wave_z must be identical (no leakage from the future)
    extended = dict(base_counts)
    extended['2021-04-15'] = {0: 100, 1: 50}
    ids_df2, labels2 = _wave_inputs(extended, K=2)
    wave2 = geometry._wave_z(ids_df2, labels2)
    df2 = pd.DataFrame({'eom': ids_df2['eom'], 'cluster': labels2, 'wave': wave2})

    for m in ['2021-01-31', '2021-02-28', '2021-03-31']:
        for c in [0, 1]:
            v1 = df[(df['eom'] == pd.Timestamp(m)) & (df['cluster'] == c)]['wave']
            v2 = df2[(df2['eom'] == pd.Timestamp(m)) & (df2['cluster'] == c)]['wave']
            if len(v1) and len(v2):
                assert np.isclose(v1.iloc[0], v2.iloc[0])


# ---------------------------------------------------------------- 4. truncation invariance (full pipeline)
def test_truncation_invariance(monkeypatch):
    _set_geom_config(monkeypatch, K=3, pca_dims=5, fit_end='2019-12-31', lookback=6, seed=3)
    ids_df, emb = _synthetic(seed=7, n=400, D=8, n_permno=6, n_blobs=3,
                              date_start='2018-01-01', date_end='2021-12-31')

    full = geometry._build_from_embeddings(ids_df, emb)

    T = pd.Timestamp('2021-03-31')  # well after GEOM_FIT_END
    keep = (ids_df['filing_date'] <= T).to_numpy()
    trunc_ids = ids_df.loc[keep].reset_index(drop=True)
    trunc_emb = emb[keep]
    truncated = geometry._build_from_embeddings(trunc_ids, trunc_emb)

    full_T = full[full['eom'] == T].sort_values('permno').reset_index(drop=True)
    trunc_T = truncated[truncated['eom'] == T].sort_values('permno').reset_index(drop=True)

    assert len(full_T) > 0
    pd.testing.assert_frame_equal(full_T, trunc_T)


# ---------------------------------------------------------------- 5. non-filers get 0
def test_non_filers_get_zero(monkeypatch, tmp_path):
    _set_geom_config(monkeypatch, K=2, pca_dims=4, fit_end='2020-06-30', seed=1)
    ids_df, emb = _synthetic(seed=5, n=60, D=6, n_permno=3, n_blobs=2,
                              date_start='2019-01-01', date_end='2020-12-31')
    _patch_existing_embeddings(monkeypatch, tmp_path, ids_df, emb)

    filer_permno = ids_df['permno'].iloc[0]
    filer_eom = (ids_df['filing_date'].iloc[0] + pd.offsets.MonthEnd(0))
    panel = pd.DataFrame({
        'permno': [filer_permno, 999],
        'eom': [filer_eom, pd.Timestamp('2020-01-31')],
    })

    out = geometry.add_geometry_features(panel)
    fill_cols = [c for c in out.columns if c not in ('permno', 'eom')]
    assert fill_cols  # geometry columns were actually added

    non_filer_row = out[out['permno'] == 999].iloc[0]
    assert (non_filer_row[fill_cols] == 0).all()


# ---------------------------------------------------------------- 6b. NaN embedding rows dropped
def test_nan_embedding_rows_dropped_before_fit(monkeypatch):
    """Filings with empty cleaned text produce all-NaN embedding rows. PCA.fit would crash on
    NaN, so those rows must be dropped before fitting/transforming, and the result should be
    identical to building from the pre-filtered (finite-only) data."""
    _set_geom_config(monkeypatch, K=3, pca_dims=5, fit_end='2020-06-30', seed=2)
    ids_df, emb = _synthetic(seed=9, n=150, D=8, n_blobs=3)

    nan_rows = np.zeros(len(ids_df), dtype=bool)
    nan_rows[[3, 40, 100]] = True
    nan_emb = emb.copy()
    nan_emb[nan_rows] = np.nan

    result = geometry._build_from_embeddings(ids_df, nan_emb)  # must not raise

    clean_ids = ids_df.loc[~nan_rows].reset_index(drop=True)
    clean_emb = emb[~nan_rows]
    expected = geometry._build_from_embeddings(clean_ids, clean_emb)

    pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------- 6c. novelty fill: past-only
def test_novelty_fill_past_only(monkeypatch):
    """When a month has no novelty values at all, the fallback fill must use only past months'
    medians, never a later month's."""
    _set_geom_config(monkeypatch, K=2, pca_dims=3, fit_end='2019-12-31', lookback=24, seed=4)
    rng = np.random.default_rng(4)
    D = 6

    # Background fit-set, dated well before the cutoff, so PCA/KMeans fit never depends on the
    # test rows below (only transform, which is row-wise and unaffected by other rows present).
    bg_n = 30
    bg_emb = rng.normal(size=(bg_n, D)).astype(np.float32)
    bg_ids = pd.DataFrame({
        'document_id': [f'bg{i}' for i in range(bg_n)],
        'permno': rng.integers(100, 110, size=bg_n),
        'filing_date': pd.to_datetime(rng.choice(pd.date_range('2018-01-01', '2019-12-31'), size=bg_n)),
    })

    # permno 1: first filing Jan (novelty NaN), second filing Feb (novelty defined -> Feb's
    # cross-sectional median is well-defined). permno 2 (Jan) and permno 3 (Mar) each file only
    # once ever -> NaN. March therefore has no novelty values at all and must fall back to a
    # past-only expanding median of prior months' medians (here, Feb's).
    test_ids = pd.DataFrame({
        'document_id': ['p1_jan', 'p1_feb', 'p2_jan', 'p3_mar'],
        'permno':      [1, 1, 2, 3],
        'filing_date': pd.to_datetime(['2020-01-15', '2020-02-15', '2020-01-20', '2020-03-15']),
    })
    test_emb = rng.normal(size=(4, D)).astype(np.float32)

    ids_df = pd.concat([bg_ids, test_ids], ignore_index=True)
    emb = np.vstack([bg_emb, test_emb])

    base = geometry._build_from_embeddings(ids_df, emb)
    mar_eom = pd.Timestamp('2020-03-31')
    mar_before = base.loc[(base['eom'] == mar_eom) & (base['permno'] == 3), 'geo_novelty_max'].iloc[0]

    # Add a much later month (April) with wildly different novelty -> must not change March's
    # already-computed filled value, which depends only on Jan/Feb.
    extra_ids = pd.DataFrame({
        'document_id': ['p1_apr', 'p4_apr'],
        'permno':      [1, 4],
        'filing_date': pd.to_datetime(['2020-04-10', '2020-04-20']),
    })
    extra_emb = (rng.normal(size=(2, D)) * 50.0).astype(np.float32)

    ids_df2 = pd.concat([ids_df, extra_ids], ignore_index=True)
    emb2 = np.vstack([emb, extra_emb])

    after = geometry._build_from_embeddings(ids_df2, emb2)
    mar_after = after.loc[(after['eom'] == mar_eom) & (after['permno'] == 3), 'geo_novelty_max'].iloc[0]

    assert np.isclose(mar_before, mar_after)


# ---------------------------------------------------------------- 6. missing embeddings
def test_missing_embeddings_returns_none(monkeypatch, tmp_path):
    missing_emb = tmp_path / 'finbert_emb_L512.npy'
    missing_ids = tmp_path / 'finbert_emb_L512_ids.parquet'
    monkeypatch.setattr(geometry, '_emb_paths', lambda: (missing_emb, missing_ids))

    assert geometry.build_geometry_features() is None

    panel = pd.DataFrame({'permno': [1, 2], 'eom': [pd.Timestamp('2020-01-31'), pd.Timestamp('2020-02-29')]})
    out = geometry.add_geometry_features(panel)
    pd.testing.assert_frame_equal(out, panel)
