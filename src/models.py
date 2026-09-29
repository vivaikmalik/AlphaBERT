"""Schedule, baselines, specialists, gate, OOS R2 / IC (SPEC section 5).

HEADLINE: pred_ew (equal-weight blend of the six specialist z-scores). pred_ew_ret is its
return-unit counterpart -- mean of the six RAW specialist forecasts (non-filer text prediction
is NaN, treated as 0) -- used only so r2_table can report an OOS R2 for the headline method,
since pred_ew itself is in z-units (brief p.20). pred_gate (the ridge regime gate) is kept only
as an ablation/explainability exhibit -- A12/A13: ~24 monthly validation observations cannot
reliably identify the gate's 18 parameters.

CHAR_GROUPS has five characteristic groups (A9: value, momentum, quality, investment_growth,
risk_liquidity); pred_ew blends these five LightGBM specialists plus a text specialist trained
on filer rows only (A1). Specialists and lgbm_all are tuned on an inner holdout (the training
window's last 12 months) by mean monthly Spearman IC, not pooled MSE (A2/A13), then refit on the
full training window; the official 24-month valid window is touched only by the gate.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

from src import config
from src import geometry
from src import selection
from src.data import feature_columns
from src.text import TEXT_FEATURES

# ---------------------------------------------------------------- groups ---
CHAR_GROUPS = {
    "value": [
        "be_me", "at_me", "ebitda_mev", "fcf_me", "div12m_me", "debt_me",
        "bev_mev", "eqpo_me", "intrinsic_value", "ni_me", "sale_me",
        "eq_dur", "eqnpo_me", "netdebt_me", "ocf_me", "rd_me",
    ],
    "momentum": [
        "prc_highprc_252d", "resff3_12_1", "resff3_6_1", "ret_1_0",
        "ret_12_1", "ret_12_7", "ret_3_1", "ret_6_1", "ret_60_12", "ret_9_1",
        "seas_1_1an", "seas_1_1na", "seas_2_5an", "seas_2_5na",
    ],
    "quality": [
        "aliq_at", "aliq_mat", "at_be", "at_turnover", "cash_at", "cop_at",
        "cop_atl1", "dgp_dsale", "dsale_dinv", "dsale_drec", "dsale_dsga",
        "earnings_variability", "ebit_sale", "f_score", "gp_at", "gp_atl1",
        "kz_index", "mispricing_perf", "ni_ar1", "ni_be", "ni_inc8q",
        "ni_ivol", "niq_at", "niq_at_chg1", "niq_be", "niq_be_chg1",
        "niq_su", "o_score", "ocf_at", "ocf_at_chg1", "ocfq_saleq_std",
        "op_at", "op_atl1", "ope_be", "ope_bel1", "opex_at", "pi_nix",
        "qmj", "qmj_growth", "qmj_prof", "qmj_safety", "rd_sale", "rd5_at",
        "sale_emp_gr1", "saleq_su", "tangibility", "tax_gr1a", "z_score",
        "ebit_bev", "sale_bev",
    ],
    "investment_growth": [
        "at_gr1", "be_gr1a", "capex_abn", "capx_gr1", "capx_gr2", "capx_gr3",
        "coa_gr1a", "col_gr1a", "cowc_gr1a", "fnl_gr1a", "lnoa_gr1a",
        "lti_gr1a", "ncoa_gr1a", "ncol_gr1a", "nfna_gr1a", "nncoa_gr1a",
        "noa_gr1a", "sti_gr1a", "ppeinv_gr1a", "debt_gr3", "emp_gr1",
        "inv_gr1", "inv_gr1a", "sale_gr1", "sale_gr3", "saleq_gr1",
        "chcsho_12m", "dbnetis_at", "eqnetis_at", "eqnpo_12m", "netis_at",
        "oaccruals_at", "oaccruals_ni", "taccruals_at", "taccruals_ni",
        "noa_at", "mispricing_mgmt",
    ],
    "risk_liquidity": [
        "age", "ami_126d", "beta_60m", "beta_dimson_21d", "betabab_1260d",
        "betadown_252d", "bidaskhl_21d", "corr_1260d", "coskew_21d",
        "dolvol_126d", "dolvol_var_126d", "iskew_capm_21d", "iskew_ff3_21d",
        "iskew_hxz4_21d", "ivol_capm_21d", "ivol_capm_252d", "ivol_ff3_21d",
        "ivol_hxz4_21d", "market_equity", "prc", "rmax1_21d", "rmax5_21d",
        "rmax5_rvol_21d", "rskew_21d", "rvol_21d", "turnover_126d",
        "turnover_var_126d", "zero_trades_126d", "zero_trades_21d",
        "zero_trades_252d",
    ],
}


# ------------------------------------------------------------- interfaces --
_FORBIDDEN = {"stock_exret", "target_month", "ret_exc_lead1m"}


def _assert_no_leakage(cols):
    bad = _FORBIDDEN & set(cols)
    assert not bad, f"leakage: forbidden columns in feature set: {bad}"


def _assert_text_features_present(panel):
    item_cols = [f"item_{it.replace('.', '_')}" for it in config.KEY_ITEMS]
    required = ["n_filings", "has_filing"] + item_cols
    missing = [c for c in required if c not in panel.columns]
    assert not missing, (
        f"panel missing required text features {missing}; run text.add_text_features(panel) "
        "before models.run_all (tone_mean/tone_min/fb_neg_max stay optional)."
    )


# Candidates fit or tuned using the valid window itself (the gate is a ridge fit on labelled
# valid rows; OLS/Ridge/Lasso/ElasticNet alphas are chosen by valid MSE) -- HEADLINE_CANDIDATES
# must exclude all of these, or pred_auto's walk-forward model choice (picked by valid IC) would
# not be genuinely out-of-sample on valid (A12).
_VALID_FIT_OR_TUNED = {"pred_gate", "pred_gate_notext", "pred_ols", "pred_ridge", "pred_lasso", "pred_enet"}


def _assert_headline_candidates_oos(candidates):
    bad = _VALID_FIT_OR_TUNED & set(candidates)
    assert not bad, (
        f"config.HEADLINE_CANDIDATES contains models fit/tuned on the validation window: {bad}"
    )


# ---------------------------------------------------------------- splits ---
def splits(panel, test_year):
    """Train/valid/test row masks (SPEC section 2). Train rows require a non-null
    stock_exret (a label is needed to fit). Valid and test rows do NOT: valid predictions are
    saved for every panel row in the valid months, including null-label future exits (A11); the
    gate is fit on the labelled subset of valid only (see run_all)."""
    tm = panel["target_month"]
    y = test_year
    train_start = config.FIRST_TARGET
    train_end = pd.Timestamp(year=y - 3, month=12, day=31)
    valid_start = pd.Timestamp(year=y - 2, month=1, day=31)
    valid_end = pd.Timestamp(year=y - 1, month=12, day=31)
    test_start = pd.Timestamp(year=y, month=1, day=31)
    test_end = min(pd.Timestamp(year=y, month=12, day=31), config.TEST_END)

    has_label = panel["stock_exret"].notna()
    train_mask = (tm >= train_start) & (tm <= train_end) & has_label
    valid_mask = (tm >= valid_start) & (tm <= valid_end)
    test_mask = (tm >= test_start) & (tm <= test_end)
    return train_mask, valid_mask, test_mask


def make_target(df):
    """stock_exret demeaned within target_month, then clipped at 1st/99th pctile.
    Rows with a null stock_exret stay null (mean/quantile ignore NaN, so labelled rows in the
    same target_month are unaffected by unlabelled rows sharing that month)."""
    y = df["stock_exret"]
    tm = df["target_month"]
    demeaned = y - y.groupby(tm).transform("mean")
    lo = demeaned.groupby(tm).transform(lambda s: s.quantile(0.01))
    hi = demeaned.groupby(tm).transform(lambda s: s.quantile(0.99))
    return demeaned.clip(lower=lo, upper=hi)


# -------------------------------------------------------------- baselines --
RIDGE_ALPHAS = np.logspace(-3, 8, 23)
LASSO_ALPHAS = np.logspace(-8, -1, 15)
ENET_ALPHAS = np.logspace(-8, -1, 15)
MAX_ROWS_PENALIZED = 100_000  # subsample cap for the Ridge/Lasso/ElasticNet alpha search


def _select_alpha(model_cls, Xtr, ytr, Xva, yva, alphas, max_rows=None, **kwargs):
    """Search alphas on the (possibly subsampled) training window, scored on valid MSE.
    Returns the best alpha only -- the caller refits at that alpha on the FULL training window,
    so subsampling the search never changes what gets fit."""
    Xs, ys = Xtr, ytr
    if max_rows is not None and len(Xtr) > max_rows:
        rng = np.random.RandomState(config.SEED)
        idx = rng.choice(len(Xtr), max_rows, replace=False)
        Xs, ys = Xtr[idx], ytr[idx]
    best_mse, best_alpha = np.inf, alphas[0]
    for a in alphas:
        m = model_cls(alpha=a, **kwargs)
        m.fit(Xs, ys)
        mse = mean_squared_error(yva, m.predict(Xva))
        if mse < best_mse:
            best_mse, best_alpha = mse, a
    return best_alpha


def _warn_if_boundary(name, alpha, alphas):
    if alpha == alphas[0] or alpha == alphas[-1]:
        print(f"WARNING: [models.fit_baselines] {name} alpha={alpha:g} is at the grid boundary "
              f"({alphas[0]:g}..{alphas[-1]:g}); widen the grid.")


# (model class, alpha grid, extra kwargs, subsample cap for the alpha search)
_PENALIZED_SPECS = [
    ("ridge", Ridge, RIDGE_ALPHAS, {}, MAX_ROWS_PENALIZED),
    ("lasso", Lasso, LASSO_ALPHAS, {"max_iter": 5000}, MAX_ROWS_PENALIZED),
    ("enet", ElasticNet, ENET_ALPHAS, {"max_iter": 5000, "l1_ratio": 0.5}, MAX_ROWS_PENALIZED),
]


def fit_baselines(Xtr, ytr, Xva, yva):
    """OLS, Ridge, Lasso, ElasticNet. Alpha is chosen on labelled-valid MSE (the search is
    subsampled to MAX_ROWS_PENALIZED rows for Ridge/Lasso/ElasticNet), then each model is refit on
    the FULL training window at its chosen alpha. These are template baselines (not gate inputs),
    so using the official valid window for alpha selection is fine."""
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xva = np.asarray(Xva, dtype=np.float32)
    out = {"ols": LinearRegression().fit(Xtr, ytr)}
    chosen = {}
    for name, cls, alphas, kwargs, max_rows in _PENALIZED_SPECS:
        alpha = _select_alpha(cls, Xtr, ytr, Xva, yva, alphas, max_rows=max_rows, **kwargs)
        _warn_if_boundary(name, alpha, alphas)
        out[name] = cls(alpha=alpha, **kwargs).fit(Xtr, ytr)
        chosen[name] = alpha

    print("[models.fit_baselines] chosen alphas: "
          + " ".join(f"{name}={alpha:g}" for name, alpha in chosen.items()))
    return out


# ------------------------------------------------------------------ lgbm ---
def _lgbm_base_params():
    """Read config.SEED/N_JOBS at call time (not import time) so tests can
    monkeypatch them, e.g. to force single-threaded determinism."""
    return dict(
        objective="regression", learning_rate=0.05, min_child_samples=1000,
        feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=10, seed=config.SEED, n_jobs=config.N_JOBS, verbose=-1,
    )


# Fixed rounds grid for IC-based selection (A13): floor 100, up to 1000.
ROUND_GRID = (100, 200, 300, 500, 700, 1000)
INNER_HOLDOUT_MONTHS = 12


def _ic_mean(pred, y, months):
    """Mean Spearman IC of pred vs y, grouped by month (reuses monthly_ic's groupby/spearman
    logic instead of a second copy)."""
    df = pd.DataFrame({"_pred": np.asarray(pred, dtype=float),
                        "stock_exret": np.asarray(y, dtype=float), "eom": months})
    return monthly_ic(df, "_pred").mean()


def _fit_lgbm_tuned(Xtr, ytr, Xva, yva, months_va):
    """Small grid over num_leaves in {7,15}; NO early stopping / MSE selection. Each num_leaves is
    trained for max(ROUND_GRID) rounds; (num_leaves, rounds) is picked by mean monthly Spearman IC
    on the holdout, evaluated at each ROUND_GRID checkpoint via predict(num_iteration=k) (A13:
    pooled-MSE early stopping was found to select degenerate 1-tree models)."""
    dtr = lgb.Dataset(Xtr, label=ytr)
    base_params = _lgbm_base_params()
    best_ic, best_leaves, best_rounds = -np.inf, ROUND_GRID[0], ROUND_GRID[0]
    for num_leaves in (7, 15):
        booster = lgb.train({**base_params, "num_leaves": num_leaves}, dtr,
                             num_boost_round=max(ROUND_GRID))
        for k in ROUND_GRID:
            pred = booster.predict(Xva, num_iteration=k)
            ic = _ic_mean(pred, yva, months_va)
            if ic > best_ic:
                best_ic, best_leaves, best_rounds = ic, num_leaves, k
    return best_leaves, best_rounds, best_ic


def _inner_holdout_mask(target_month):
    """Boolean array, True for rows in the last INNER_HOLDOUT_MONTHS target months of the given
    (train) target_month values. Shared by _fit_specialist, which tunes num_leaves/rounds on this
    holdout, and run_all's feature-selection call, which must EXCLUDE it -- selection must never
    see the months later used to tune the specialists, or its inner-holdout IC would be
    optimistic."""
    tm = np.asarray(target_month)
    months = np.sort(np.unique(tm))
    ho = min(INNER_HOLDOUT_MONTHS, max(1, len(months) - 1))
    return np.isin(tm, months[-ho:])


def _fit_specialist(train_df, y_train, feature_cols, label=None):
    """Tune (num_leaves, rounds) on an inner holdout = the last INNER_HOLDOUT_MONTHS target
    months of train, then refit on the FULL training window. Never touches the official valid
    window, so specialist forecasts on valid are genuinely OOS (A2). If `label` is given, prints
    the chosen num_leaves/rounds and inner-holdout IC (A13)."""
    tm = train_df["target_month"].values
    is_ho = _inner_holdout_mask(tm)
    X = train_df[feature_cols].astype(np.float32).values
    num_leaves, num_rounds, ic = _fit_lgbm_tuned(
        X[~is_ho], y_train[~is_ho], X[is_ho], y_train[is_ho], tm[is_ho])
    if label is not None:
        print(f"[models._fit_specialist] {label}: num_leaves={num_leaves} rounds={num_rounds} "
              f"inner_holdout_ic={ic:.4f}")
    dfull = lgb.Dataset(X, label=y_train)
    return lgb.train({**_lgbm_base_params(), "num_leaves": num_leaves}, dfull, num_boost_round=num_rounds)


def _predict_both(model, Xva, Xte):
    """The repeated {'valid': ..., 'test': ...} prediction dict, for any fitted model exposing
    .predict (linear baselines, LightGBM boosters, the ridge gate)."""
    return {"valid": model.predict(Xva), "test": model.predict(Xte)}


def _predict_filer_only(model, df, filer_mask, cols):
    """Raw prediction array (len(df),), NaN for the non-filer rows the text specialist never
    sees (A1). _zscore_by_eom turns that NaN into an exact Z=0; pred_ew_ret treats it as 0 too."""
    out = np.full(len(df), np.nan)
    out[filer_mask.values] = model.predict(df.loc[filer_mask, cols].astype(np.float32).values)
    return out


# ------------------------------------------------------------------ gate ---
def _zscore_by_eom(values, eom):
    """Z-score within eom. NaN inputs (e.g. the text specialist's non-filer rows) are ignored by
    the group mean/std, then fillna(0.0) sets their own z-score to exactly 0 -- i.e. this already
    implements a filer-only z-score with a neutral 0 for non-filers; no separate helper needed."""
    s = pd.Series(np.asarray(values, dtype=float), index=eom.index)
    g = s.groupby(eom)
    mean = g.transform("mean")
    std = g.transform("std")
    z = (s - mean) / std.replace(0.0, np.nan)
    return z.fillna(0.0)


def _join_state(df, state):
    s = state.reindex(df["eom"].values)[list(config.STATE_VARS)].reset_index(drop=True)
    s.index = df.index
    return s


def _build_gate_design(Z, S):
    cols = {k: Z[k].values for k in Z.columns}
    for k in Z.columns:
        for j in S.columns:
            cols[f"{k}__{j}"] = Z[k].values * S[j].values
    return pd.DataFrame(cols, index=Z.index)


def _gate_alphas(n):
    """Ridge alpha grid scaled by n (A13): n * logspace(-3, 2, 11). sklearn's Ridge objective is
    ||y - Xw||^2 + alpha*||w||^2 (not divided by n), so the effective penalty scales with the
    number of rows; scaling the grid by n keeps it comparable across test years with very
    different valid-row counts."""
    return n * np.logspace(-3, 2, 11)


def fit_gate(Z_valid, S_valid, y_valid, eom_valid):
    """Ridge gate fit on labelled valid rows; alpha chosen by RidgeCV with folds grouped by eom
    (whole months, not individual rows). The valid window is always >= 24 target months (SPEC
    section 2), so GroupKFold always has >= 2 groups to split."""
    Xg = _build_gate_design(Z_valid, S_valid)
    y = np.asarray(y_valid, dtype=float)
    alphas = _gate_alphas(len(y))
    groups = eom_valid.values
    n_groups = len(np.unique(groups))
    cv = list(GroupKFold(n_splits=min(6, n_groups)).split(Xg.values, y, groups=groups))
    model = RidgeCV(alphas=alphas, fit_intercept=False, cv=cv)
    model.fit(Xg.values, y)
    return model, list(Xg.columns)


def gate_coefs_frame(model, columns, test_year, mu, sd):
    rows = []
    for col, coef in zip(columns, model.coef_):
        spec, term = col.split("__", 1) if "__" in col else (col, "base")
        row = {"test_year": test_year, "specialist": spec, "term": term, "coef": coef}
        for v in config.STATE_VARS:
            row[f"state_mean_{v}"] = mu[v]
            row[f"state_std_{v}"] = sd[v]
        rows.append(row)
    return pd.DataFrame(rows)


def _fallback_group_cols(report_df, group_cols, min_per_group):
    """Used when feature selection drops every one of a char group's columns: falls back to that
    group's top `min_per_group` candidate features by |ic_t|. Relies on `report_df` (from
    selection.select_features) carrying an 'ic_t' column for every candidate feature it considered
    -- selected or not -- keyed by 'feature', so this fallback always has something to rank from."""
    cand = report_df[report_df["feature"].isin(group_cols)].copy()
    cand["_abs_ic_t"] = cand["ic_t"].abs()
    return cand.sort_values("_abs_ic_t", ascending=False)["feature"].head(min_per_group).tolist()


# --------------------------------------------------------------- run_all ---
PRED_COLS = [
    "pred_ols", "pred_ridge", "pred_lasso", "pred_enet", "pred_lgbm_all",
    "pred_spec_value", "pred_spec_momentum", "pred_spec_quality",
    "pred_spec_investment_growth", "pred_spec_risk_liquidity", "pred_spec_text",
    "pred_gate", "pred_gate_notext", "pred_ew_ret", "pred_ew", "pred_ew_notext",
    "pred_blend", "pred_auto",
]
# Return-unit forecasts get an OOS R2 in r2_table; pred_ew/pred_ew_notext blend z-scores (not
# returns), pred_blend blends two z-scores, and pred_auto is whichever HEADLINE_CANDIDATES column
# was picked (itself z-valued or z-blended) -- so none of the four have a meaningful R2, IC only.
R2_COLS = [c for c in PRED_COLS if c not in ("pred_ew", "pred_ew_notext", "pred_blend", "pred_auto")]


def run_all(panel, state, cache_dir=None):
    cache_dir = config.CACHE_DIR if cache_dir is None else Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _assert_text_features_present(panel)
    _assert_headline_candidates_oos(config.HEADLINE_CANDIDATES)
    feature_cols = feature_columns(panel)
    text_cols = [c for c in list(TEXT_FEATURES) + list(geometry.GEOMETRY_FEATURES)
                 if c in panel.columns]
    _assert_no_leakage(feature_cols)
    _assert_no_leakage(text_cols)

    char_to_group = {c: g for g, cs in CHAR_GROUPS.items() for c in cs}
    group_cols = {
        g: [c for c in feature_cols if char_to_group.get(c[5:] if c.startswith("miss_") else c) == g]
        for g in CHAR_GROUPS
    }
    specialists_all = list(CHAR_GROUPS) + ["text"]

    all_frames, all_gate, all_selected, all_headline = [], [], [], []
    for test_year in config.TEST_YEARS:
        ty0 = time.time()
        train_mask, valid_mask, test_mask = splits(panel, test_year)
        train, valid, test = panel.loc[train_mask], panel.loc[valid_mask], panel.loc[test_mask]
        valid_lab = valid["stock_exret"].notna()

        ytr = make_target(train).values
        yva_full = make_target(valid)  # NaN for unlabelled valid rows, by construction
        yva_lab = yva_full.loc[valid_lab].values

        # Feature selection (per test year, TRAINING rows only -- walk-forward, never touches
        # valid/test). Also excludes train's inner-holdout months (the last INNER_HOLDOUT_MONTHS
        # target months), since those are later used to tune each specialist's num_leaves/rounds
        # in _fit_specialist -- selection must not see them, or its report's IC would be
        # optimistic for the very rows tuning is scored on. report_df must cover every candidate
        # feature (selected or not) with an 'ic_t' column, so the per-group fallback below always
        # has something to rank.
        # A16: when FEATURE_SELECTION is False but SELECTION_REPORT is True, the report is still
        # computed and written (selected_features.csv) for inspection, but every model below is
        # fit on the FULL feature set (report_cols is not used for fitting in that case).
        want_report = config.FEATURE_SELECTION or getattr(config, "SELECTION_REPORT", True)
        if want_report:
            is_ho = _inner_holdout_mask(train["target_month"].values)
            report_cols, sel_report = selection.select_features(
                train.loc[~is_ho], ytr[~is_ho], feature_cols, CHAR_GROUPS)
            sel_report = sel_report.copy()
            sel_report["test_year"] = test_year
            all_selected.append(sel_report)
        else:
            report_cols, sel_report = feature_cols, None
        sel_cols = report_cols if config.FEATURE_SELECTION else feature_cols

        preds = {}  # name -> {'valid': arr (all valid rows), 'test': arr (all test rows)}
        Xva_all = valid[sel_cols].astype(np.float32).values
        Xva_lab = Xva_all[valid_lab.values]
        Xte_all = test[sel_cols].astype(np.float32).values

        base = fit_baselines(train[sel_cols].astype(np.float32).values, ytr, Xva_lab, yva_lab)
        for name, m in base.items():
            preds[f"pred_{name}"] = _predict_both(m, Xva_all, Xte_all)

        lgbm_all = _fit_specialist(train, ytr, sel_cols, label=f"lgbm_all y={test_year}")
        preds["pred_lgbm_all"] = _predict_both(lgbm_all, Xva_all, Xte_all)

        for g in CHAR_GROUPS:
            cols = group_cols[g]
            if config.FEATURE_SELECTION:
                sel_set = set(sel_cols)
                cols = [c for c in cols if c in sel_set]
                if not cols:
                    cols = _fallback_group_cols(sel_report, group_cols[g], config.SELECTION_MIN_PER_GROUP)
            m = _fit_specialist(train, ytr, cols, label=f"{g} y={test_year}")
            preds[f"pred_spec_{g}"] = _predict_both(
                m, valid[cols].astype(np.float32).values, test[cols].astype(np.float32).values)

        train_filer = train["has_filing"].astype(bool)
        valid_filer = valid["has_filing"].astype(bool)
        test_filer = test["has_filing"].astype(bool)
        m_text = _fit_specialist(train.loc[train_filer.values], ytr[train_filer.values], text_cols,
                                  label=f"text y={test_year}")
        preds["pred_spec_text"] = {
            "valid": _predict_filer_only(m_text, valid, valid_filer, text_cols),
            "test": _predict_filer_only(m_text, test, test_filer, text_cols),
        }

        Z_valid = pd.DataFrame({k: _zscore_by_eom(preds[f"pred_spec_{k}"]["valid"], valid["eom"])
                                 for k in specialists_all})
        Z_test = pd.DataFrame({k: _zscore_by_eom(preds[f"pred_spec_{k}"]["test"], test["eom"])
                                for k in specialists_all})

        # Headline's OOS-R2 proxy (brief p.20 / A12 disclosure): mean of the six RAW specialist
        # return-unit forecasts, non-filer text prediction (NaN) treated as 0.
        raw_specs = [preds[f"pred_spec_{k}"] for k in specialists_all]
        preds["pred_ew_ret"] = {
            split: np.mean([np.nan_to_num(r[split], nan=0.0) for r in raw_specs], axis=0)
            for split in ("valid", "test")
        }

        valid_eoms = valid["eom"].unique()
        state_valid = state.loc[state.index.isin(valid_eoms), list(config.STATE_VARS)]
        mu, sd_raw = state_valid.mean(), state_valid.std()
        sd = sd_raw.replace(0.0, np.nan)
        S_valid = (_join_state(valid, state) - mu) / sd
        S_test = (_join_state(test, state) - mu) / sd
        assert S_valid.notna().all().all(), f"missing/degenerate market state for valid eoms, test_year={test_year}"
        assert S_test.notna().all().all(), f"missing/degenerate market state for test eoms, test_year={test_year}"

        Z_valid_lab, S_valid_lab = Z_valid.loc[valid_lab], S_valid.loc[valid_lab]
        eom_valid_lab = valid.loc[valid_lab, "eom"]

        gate, gate_cols = fit_gate(Z_valid_lab, S_valid_lab, yva_lab, eom_valid_lab)
        preds["pred_gate"] = _predict_both(
            gate, _build_gate_design(Z_valid, S_valid).values, _build_gate_design(Z_test, S_test).values)
        all_gate.append(gate_coefs_frame(gate, gate_cols, test_year, mu, sd_raw))

        nt = list(CHAR_GROUPS)
        gate_nt, _ = fit_gate(Z_valid_lab[nt], S_valid_lab, yva_lab, eom_valid_lab)
        preds["pred_gate_notext"] = _predict_both(
            gate_nt, _build_gate_design(Z_valid[nt], S_valid).values,
            _build_gate_design(Z_test[nt], S_test).values)

        preds["pred_ew"] = {"valid": Z_valid[specialists_all].mean(axis=1).values,
                             "test": Z_test[specialists_all].mean(axis=1).values}
        preds["pred_ew_notext"] = {
            "valid": Z_valid[nt].mean(axis=1).values,
            "test": Z_test[nt].mean(axis=1).values,
        }

        # pred_blend: mean of within-eom z-scores of pred_ew and pred_lgbm_all.
        preds["pred_blend"] = {
            split: (_zscore_by_eom(preds["pred_ew"][split], df["eom"]).values
                    + _zscore_by_eom(preds["pred_lgbm_all"][split], df["eom"]).values) / 2.0
            for split, df in (("valid", valid), ("test", test))
        }

        # pred_auto: per test year, the HEADLINE_CANDIDATES column with the best mean monthly
        # Spearman IC on THAT year's own labelled valid rows -- walk-forward model choice, never
        # touches test rows/labels.
        valid_ic_df = pd.DataFrame({"eom": eom_valid_lab.values,
                                     "stock_exret": valid.loc[valid_lab, "stock_exret"].values})
        cand_ic = {}
        for cand in config.HEADLINE_CANDIDATES:
            valid_ic_df["_cand"] = preds[cand]["valid"][valid_lab.values]
            cand_ic[cand] = monthly_ic(valid_ic_df, "_cand").mean()
        chosen = max(cand_ic, key=cand_ic.get)
        all_headline.append(pd.DataFrame({
            "test_year": test_year, "candidate": list(cand_ic.keys()),
            "valid_mean_ic": list(cand_ic.values()),
            "chosen": [c == chosen for c in cand_ic],
        }))
        preds["pred_auto"] = {"valid": preds[chosen]["valid"], "test": preds[chosen]["test"]}

        def _mk(df, split_label):
            out = pd.DataFrame({
                "permno": df["permno"].values, "eom": df["eom"].values,
                "target_month": df["target_month"].values,
                "stock_exret": df["stock_exret"].values,
                "test_year": test_year, "split": split_label,
            })
            for name in PRED_COLS:
                out[name] = preds[name][split_label]
            return out

        frames = [_mk(test, "test"), _mk(valid, "valid")]  # all valid rows, incl. null-label (A11)
        all_frames.append(pd.concat(frames, ignore_index=True))
        print(f"[models.run_all] test_year={test_year} train={len(train)} "
              f"valid={len(valid)} (labelled={int(valid_lab.sum())}) test={len(test)} "
              f"({time.time() - ty0:.1f}s)")

    preds_df = pd.concat(all_frames, ignore_index=True)
    gate_df = pd.concat(all_gate, ignore_index=True)
    preds_df.to_parquet(cache_dir / "preds.parquet")
    gate_df.to_parquet(cache_dir / "gate_coefs.parquet")
    if all_selected:
        pd.concat(all_selected, ignore_index=True).to_csv(
            config.TABLE_DIR / "selected_features.csv", index=False)
    pd.concat(all_headline, ignore_index=True).to_csv(
        config.TABLE_DIR / "headline_choice.csv", index=False)
    print(f"[models.run_all] done in {time.time() - t0:.1f}s")
    return preds_df


# ------------------------------------------------------------------ stats --
def oos_r2(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return 1 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)


def monthly_ic(df, col):
    def _ic(g):
        g = g.dropna()
        if len(g) < 5:
            return np.nan
        return spearmanr(g[col], g["stock_exret"])[0]
    return df.groupby("eom")[[col, "stock_exret"]].apply(_ic)


def r2_table(preds):
    """OOS R2 (vs raw and demeaned stock_exret) for return-unit forecasts only (R2_COLS); IC for
    every model in PRED_COLS, including ew/ew_notext (whose R2 is NaN -- they blend z-scores, not
    returns, so an R2 against stock_exret is not meaningful)."""
    df = preds[(preds["split"] == "test") & preds["stock_exret"].notna()].copy()
    df["_demeaned"] = df["stock_exret"] - df.groupby("eom")["stock_exret"].transform("mean")
    rows = []
    for col in PRED_COLS:
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        ic = monthly_ic(sub, col)
        n = ic.count()
        t_ic = (ic.mean() / (ic.std(ddof=1) / np.sqrt(n))) if n > 1 else np.nan
        if col in R2_COLS:
            r2 = oos_r2(sub["stock_exret"], sub[col])
            r2_demeaned = oos_r2(sub["_demeaned"], sub[col])
        else:
            r2 = r2_demeaned = np.nan
        rows.append({
            "model": col, "oos_r2": r2, "oos_r2_demeaned": r2_demeaned,
            "mean_ic": ic.mean(), "ic_tstat": t_ic,
        })
    return pd.DataFrame(rows)
