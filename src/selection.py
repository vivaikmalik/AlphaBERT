"""Per-training-window factor selection (SPEC: "which stock-level factors are most predictive").

Two independent screens on characteristic features (never miss_<c> flags themselves):
  1. Monthly Spearman IC vs y within each target_month of the training rows passed in; a
     t-stat across months, Benjamini-Hochberg FDR control at config.SELECTION_FDR_Q.
  2. Stability-selection Lasso: config.SELECTION_LASSO_REPS subsamples of 50% of the training
     months (seeded config.SEED, rows capped at ~50k), LassoCV per subsample; selection
     frequency = share of subsamples with a nonzero coefficient.
selected = BH survivors UNION (lasso_freq >= config.SELECTION_LASSO_FREQ), then each group is
topped up to config.SELECTION_MIN_PER_GROUP by its strongest remaining |t| chars. miss_<c> flags
carry no stats of their own -- they are selected iff <c> is selected.

Only the train_df/y/feature_cols/groups passed in are ever touched (no panel-level or global
state besides config), so results are deterministic and reusable across independently-run
training windows.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LassoCV

from src import config

MONTH_COL = "target_month"
MIN_ROWS_PER_MONTH = 5
LASSO_ROW_CAP = 50_000


def _prep(train_df, y, char_feats):
    """Rows with a labelled y, as plain numpy arrays. Features are assumed already
    cross-sectionally ranked/scaled (as src/data.py's panel features are); any remaining NaN is
    filled with 0.0, matching that convention."""
    y = pd.Series(np.asarray(y, dtype=float), index=train_df.index)
    ok = y.notna()
    months = train_df.loc[ok, MONTH_COL].to_numpy()
    X = train_df.loc[ok, char_feats].astype(float).fillna(0.0).to_numpy()
    yv = y.loc[ok].to_numpy()
    return months, X, yv


def _monthly_ic_stats(months, X, yv):
    """Per-feature mean/t-stat/p-value of the monthly Spearman IC vs yv, one value per column of
    X. Ranks (and the correlation itself) are computed for every feature at once per month via
    plain numpy, rather than calling scipy.stats.spearmanr per (feature, month) pair."""
    n_features = X.shape[1]
    uniq = np.unique(months)
    ics = []
    for m in uniq:
        mask = months == m
        if mask.sum() < MIN_ROWS_PER_MONTH:
            continue
        Xr = pd.DataFrame(X[mask]).rank(method="average").to_numpy()
        yr = pd.Series(yv[mask]).rank(method="average").to_numpy()
        Xr = Xr - Xr.mean(axis=0)
        yr = yr - yr.mean()
        denom = np.sqrt((Xr ** 2).sum(axis=0) * (yr ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            ics.append((Xr.T @ yr) / denom)

    if not ics:
        nan = np.full(n_features, np.nan)
        return nan, nan.copy(), nan.copy()

    ic_arr = np.array(ics)  # (n_valid_months, n_features)
    n_months = np.sum(~np.isnan(ic_arr), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic_mean = np.nanmean(ic_arr, axis=0)
        ic_std = np.nanstd(ic_arr, axis=0, ddof=1)
        t = ic_mean / (ic_std / np.sqrt(n_months))
        p = 2 * (1 - norm.cdf(np.abs(t)))
    return ic_mean, t, p


def _bh_reject(pvals, q):
    """Standard Benjamini-Hochberg step-up procedure. NaN p-values never reject."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    filled = np.where(np.isnan(pvals), np.inf, pvals)
    order = np.argsort(filled)
    ranked = filled[order]
    thresh = (np.arange(1, n + 1) / n) * q
    ok = np.isfinite(ranked) & (ranked <= thresh)
    reject = np.zeros(n, dtype=bool)
    if ok.any():
        k = int(np.max(np.where(ok)[0]))
        reject[order[: k + 1]] = True
    return reject


def _lasso_stability(months, X, yv, n_features):
    """Selection frequency per feature over config.SELECTION_LASSO_REPS subsamples of 50% of the
    distinct training months (rows capped at ~LASSO_ROW_CAP), seeded off config.SEED so repeated
    calls on the same rows are deterministic."""
    uniq_months = np.unique(months)
    n_half = max(1, len(uniq_months) // 2)
    rng = np.random.RandomState(config.SEED)
    hits = np.zeros(n_features)
    reps = config.SELECTION_LASSO_REPS
    for _ in range(reps):
        sel_months = rng.choice(uniq_months, size=n_half, replace=False)
        idx = np.where(np.isin(months, sel_months))[0]
        if len(idx) > LASSO_ROW_CAP:
            idx = rng.choice(idx, size=LASSO_ROW_CAP, replace=False)
        Xs, ys = X[idx], yv[idx]
        model = LassoCV(cv=3, n_alphas=20, max_iter=2000, random_state=config.SEED)
        model.fit(Xs, ys)
        hits += np.abs(model.coef_) > 1e-10
    return hits / reps


def _fill_min_per_group(selected, ic_t, char_feats, groups):
    """Top up each group to config.SELECTION_MIN_PER_GROUP with its strongest remaining |t|
    characteristics (NaN t-stats sort last, i.e. lowest priority)."""
    idx = {c: i for i, c in enumerate(char_feats)}
    selected = selected.copy()
    for chars in groups.values():
        in_scope = [c for c in chars if c in idx]
        have = sum(selected[idx[c]] for c in in_scope)
        need = config.SELECTION_MIN_PER_GROUP - have
        if need <= 0:
            continue
        candidates = [c for c in in_scope if not selected[idx[c]]]
        candidates.sort(key=lambda c: np.inf if np.isnan(ic_t[idx[c]]) else -abs(ic_t[idx[c]]))
        for c in candidates[:need]:
            selected[idx[c]] = True
    return selected


def select_features(train_df, y, feature_cols, groups):
    """(selected_cols, report_df) for one training window, using only train_df/y rows passed in.

    train_df: training rows, must include MONTH_COL ('target_month').
    y: target aligned with train_df (row order/index), e.g. models.make_target(train_df).
    feature_cols: candidate feature columns (characteristics + their miss_<c> flags).
    groups: dict group -> list of characteristic names (e.g. models.CHAR_GROUPS).

    report_df columns: feature, group, ic_mean, ic_t, p_value, bh_pass, lasso_freq, selected.
    miss_<c> rows carry NaN stats/bh_pass=False and selected == the selection status of <c>.
    """
    char_feats = [c for c in feature_cols if not c.startswith("miss_")]
    miss_feats = [c for c in feature_cols if c.startswith("miss_")]
    char_to_group = {c: g for g, cs in groups.items() for c in cs}

    months, X, yv = _prep(train_df, y, char_feats)
    ic_mean, ic_t, p_value = _monthly_ic_stats(months, X, yv)
    bh_pass = _bh_reject(p_value, config.SELECTION_FDR_Q)
    lasso_freq = _lasso_stability(months, X, yv, len(char_feats))

    selected = bh_pass | (lasso_freq >= config.SELECTION_LASSO_FREQ)
    selected = _fill_min_per_group(selected, ic_t, char_feats, groups)
    char_selected = dict(zip(char_feats, selected))

    rows = [{
        "feature": c, "group": char_to_group.get(c), "ic_mean": ic_mean[i], "ic_t": ic_t[i],
        "p_value": p_value[i], "bh_pass": bool(bh_pass[i]), "lasso_freq": lasso_freq[i],
        "selected": bool(selected[i]),
    } for i, c in enumerate(char_feats)]

    for m in miss_feats:
        base = m[len("miss_"):]
        rows.append({
            "feature": m, "group": char_to_group.get(base), "ic_mean": np.nan, "ic_t": np.nan,
            "p_value": np.nan, "bh_pass": False, "lasso_freq": np.nan,
            "selected": bool(char_selected.get(base, False)),
        })

    report_df = pd.DataFrame(rows)
    keep = dict(zip(report_df["feature"], report_df["selected"]))
    selected_cols = [c for c in feature_cols if keep.get(c, False)]
    return selected_cols, report_df
