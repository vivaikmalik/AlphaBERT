"""FinBERT embedding "event geometry" features: how unusual a filing's disclosure is relative to
its own firm's history and to the cross-section of other filings. See docs/SPEC.md.

Everything here is strictly point-in-time: the PCA whitening and the k-means event-type clusters
are fitted ONLY on filings dated on or before config.GEOM_FIT_END, then applied (transform/
predict, never re-fit) to every filing. Per-firm novelty only looks at that same firm's strictly
earlier filings within a lookback window, and the event-wave z-score only looks at past months.

Pipeline per filing:
  1. PCA(config.GEOM_PCA_DIMS, whiten=True) fit on pre-cutoff embeddings; transform all; L2-normalize.
  2. KMeans(config.GEOM_K) fit on the same pre-cutoff set; every filing assigned to its nearest
     centroid. atypicality = distance to own centroid / that cluster's fit-set median distance
     ("spread").
  3. novelty = 1 - max cosine similarity to the same permno's strictly-earlier filings within
     config.GEOM_NOVELTY_LOOKBACK_MONTHS (NaN if none).
  4. wave_z: per (month, cluster) filing count vs. the trailing 12-month mean count for that
     cluster, using past months only.
  5. signed_atyp = atypicality * tone (tone = fb_pos - fb_neg; 0 if no FinBERT scores cached).

Aggregated to (permno, eom=month-end of filing_date).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src import config, text

GEOMETRY_FEATURES = (
    ['geo_atyp_max', 'geo_atyp_mean', 'geo_novelty_max', 'geo_signed_atyp_min', 'geo_wave_max']
    + [f'geo_c{k}' for k in range(config.GEOM_K)]
)


def _emb_paths():
    """Read config.CACHE_DIR at call time (not import time) so tests can monkeypatch it."""
    return config.CACHE_DIR / 'finbert_emb_L512.npy', config.CACHE_DIR / 'finbert_emb_L512_ids.parquet'


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def _fit(ids_df: pd.DataFrame, emb: np.ndarray):
    """Fit PCA+KMeans on filings dated <= config.GEOM_FIT_END only. Returns (pca, kmeans, X) where
    X is the L2-normalized whitened-PCA embedding of every row in ids_df (fit set and beyond)."""
    fit_mask = (pd.to_datetime(ids_df['filing_date']) <= config.GEOM_FIT_END).to_numpy()
    emb = np.asarray(emb, dtype=np.float32)

    pca = PCA(n_components=config.GEOM_PCA_DIMS, whiten=True, random_state=config.SEED)
    pca.fit(emb[fit_mask])
    X = _l2_normalize(pca.transform(emb))

    kmeans = KMeans(n_clusters=config.GEOM_K, n_init=4, random_state=config.SEED)
    kmeans.fit(X[fit_mask])
    return pca, kmeans, X


def _cluster_assign(kmeans: KMeans, X: np.ndarray):
    """Nearest-centroid label and Euclidean distance to own centroid, for every row."""
    labels = kmeans.predict(X)
    dist = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)
    return labels, dist


def _cluster_spreads(labels: np.ndarray, dist: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Median distance of fit-set members to their own centroid, per cluster. Empty/degenerate
    clusters fall back to the median spread across clusters (or a tiny epsilon), so atypicality
    never divides by zero."""
    fit_labels, fit_dist = labels[fit_mask], dist[fit_mask]
    spreads = np.full(config.GEOM_K, np.nan)
    for c in range(config.GEOM_K):
        d = fit_dist[fit_labels == c]
        if len(d):
            spreads[c] = np.median(d)
    fallback = np.nanmedian(spreads) if np.isfinite(spreads).any() else 1e-6
    fallback = fallback if fallback > 0 else 1e-6
    return np.where(np.isfinite(spreads) & (spreads > 0), spreads, fallback)


def _novelty(ids_df: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    """1 - max cosine similarity (dot product; X is L2-normalized) to the SAME permno's filings
    with filing_date strictly earlier and within GEOM_NOVELTY_LOOKBACK_MONTHS. NaN if none."""
    lookback = pd.DateOffset(months=config.GEOM_NOVELTY_LOOKBACK_MONTHS)
    dates_all = pd.to_datetime(ids_df['filing_date']).to_numpy()
    novelty = np.full(len(ids_df), np.nan, dtype=np.float64)

    for _, idx in ids_df.groupby('permno').indices.items():
        order = idx[np.argsort(dates_all[idx], kind='mergesort')]
        dates = pd.DatetimeIndex(dates_all[order])
        vecs = X[order]
        sims = vecs @ vecs.T
        for pos in range(len(order)):
            cutoff = dates[pos] - lookback
            mask = (dates < dates[pos]) & (dates >= cutoff)
            if mask.any():
                novelty[order[pos]] = 1.0 - sims[pos, mask].max()
    return novelty


def _wave_z(ids_df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Per (month, cluster): count vs. trailing-12-month mean count for that cluster, PAST months
    only. wave_z = (count - base) / sqrt(base + 1)."""
    K = config.GEOM_K
    df = pd.DataFrame({'eom': ids_df['eom'].values, 'cluster': labels})

    counts = df.groupby(['eom', 'cluster']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(K), fill_value=0)

    full_periods = pd.period_range(df['eom'].min().to_period('M'), df['eom'].max().to_period('M'), freq='M')
    full_eoms = full_periods.to_timestamp(how='end').normalize()
    counts = counts.reindex(full_eoms, fill_value=0)
    counts.index.name = 'eom'
    counts.columns.name = 'cluster'

    base = counts.shift(1).rolling(12, min_periods=1).mean().fillna(0.0)
    wave = (counts - base) / np.sqrt(base + 1)

    wave_long = wave.stack().rename('wave_z').reset_index()
    merged = df.merge(wave_long, on=['eom', 'cluster'], how='left')
    return merged['wave_z'].to_numpy()


def _tone(ids_df: pd.DataFrame) -> np.ndarray:
    """fb_pos - fb_neg per document_id from the cached FinBERT scores; 0 if that document has no
    score or the scores file doesn't exist at all."""
    scores_path = text.scores_path_for(text.MAX_LENGTH)
    if not scores_path.exists():
        return np.zeros(len(ids_df), dtype=np.float64)
    scores = pd.read_parquet(scores_path, columns=['document_id', 'fb_pos', 'fb_neg'])
    tone_map = (scores['fb_pos'] - scores['fb_neg'])
    tone_map.index = scores['document_id']
    return ids_df['document_id'].map(tone_map).fillna(0.0).to_numpy()


def _build_from_embeddings(ids_df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    ids_df = ids_df.reset_index(drop=True).copy()
    emb = np.asarray(emb, dtype=np.float32)
    # Filings whose cleaned text was empty get an all-NaN embedding row. Drop them before
    # fitting/transforming: PCA.fit crashes on NaN, and these filings simply get no geometry
    # features (0 after the join in add_geometry_features), same as non-filers.
    finite = np.isfinite(emb).all(axis=1)
    if not finite.all():
        ids_df = ids_df.loc[finite].reset_index(drop=True)
        emb = emb[finite]

    ids_df['filing_date'] = pd.to_datetime(ids_df['filing_date']).astype('datetime64[ns]')
    ids_df['eom'] = ids_df['filing_date'] + pd.offsets.MonthEnd(0)
    fit_mask = (ids_df['filing_date'] <= config.GEOM_FIT_END).to_numpy()

    pca, kmeans, X = _fit(ids_df, emb)
    labels, dist = _cluster_assign(kmeans, X)
    spreads = _cluster_spreads(labels, dist, fit_mask)
    atyp = dist / spreads[labels]

    novelty = _novelty(ids_df, X)
    tone = _tone(ids_df)
    signed_atyp = atyp * tone
    wave = _wave_z(ids_df, labels)

    long = pd.DataFrame({
        'permno': ids_df['permno'].values,
        'eom': ids_df['eom'].values,
        'cluster': labels,
        'atyp': atyp,
        'novelty': novelty,
        'signed_atyp': signed_atyp,
        'wave': wave,
    })

    agg = long.groupby(['permno', 'eom']).agg(
        geo_atyp_max=('atyp', 'max'),
        geo_atyp_mean=('atyp', 'mean'),
        geo_novelty_max=('novelty', 'max'),
        geo_signed_atyp_min=('signed_atyp', 'min'),
        geo_wave_max=('wave', 'max'),
    ).reset_index()

    counts = long.groupby(['permno', 'eom', 'cluster']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(config.GEOM_K), fill_value=0)
    counts.columns = [f'geo_c{k}' for k in range(config.GEOM_K)]
    agg = agg.merge(counts.reset_index(), on=['permno', 'eom'], how='left')
    for k in range(config.GEOM_K):
        agg[f'geo_c{k}'] = agg[f'geo_c{k}'].fillna(0).astype('int64')

    # geo_novelty_max: NaN (firm had no eligible prior filing) -> that month's cross-sectional
    # median (over permnos that DID have one that month); a month where nobody has one falls back
    # to a PAST-ONLY expanding median of prior months' medians (never future months, to avoid
    # look-ahead), or 0 if there's no prior data either.
    month_med = agg.groupby('eom')['geo_novelty_max'].median()
    past_fallback = month_med.expanding().median().shift(1)
    month_fill = month_med.fillna(past_fallback).fillna(0.0)
    agg['geo_novelty_max'] = agg['geo_novelty_max'].fillna(agg['eom'].map(month_fill)).fillna(0.0)

    return agg.sort_values(['permno', 'eom']).reset_index(drop=True)


def build_geometry_features():
    """Per (permno, eom): geometry features built from FinBERT embeddings. Returns None (with a
    warning) if the cached embedding files don't exist yet."""
    emb_path, ids_path = _emb_paths()
    if not (emb_path.exists() and ids_path.exists()):
        print(f'build_geometry_features: WARNING embeddings not found '
              f'({emb_path.name}, {ids_path.name}) -> geometry features skipped')
        return None
    ids_df, emb = text.load_embeddings()
    return _build_from_embeddings(ids_df, emb)


def add_geometry_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Left-join geometry features onto panel (permno, eom); 0-fill stock-months with no filing
    (geometry, like text features, exists only for filers). Panel is returned unchanged if the
    embedding cache doesn't exist."""
    feat = build_geometry_features()
    if feat is None:
        return panel
    fill_cols = [c for c in feat.columns if c not in ('permno', 'eom')]
    out = panel.merge(feat, on=['permno', 'eom'], how='left')
    for c in fill_cols:
        out[c] = out[c].fillna(0)
    return out
