"""AlphaBERT -- single-file competition submission (docs/SPEC.md A6).
Bundled by build_submission_main.py; needs no src/ package. Paths (data/, outputs/) resolve
relative to env var ALPHABERT_ROOT (fallback: cwd) -- set it when running from elsewhere:
    ALPHABERT_ROOT=/path/to/AlphaBERT python MAIN.py --dry
"""
import sys, types

def _load_module(name, source, **deps):
    mod = types.ModuleType(name)
    mod.__dict__.update(deps)
    exec(compile(source, f"<bundled:{name}>", "exec"), mod.__dict__)
    sys.modules[name] = mod
    return mod

# ===== src/config.py =====
config = _load_module('config', r'''import os

import pandas as pd
from pathlib import Path

import os as _os
ROOT = Path(_os.environ.get('ALPHABERT_ROOT', '.')).resolve()
DATA_DIR = ROOT / 'data'
EXT_DIR = DATA_DIR / 'external'
OUT_DIR = ROOT / 'outputs'
CACHE_DIR = OUT_DIR / 'cache'
FIG_DIR = OUT_DIR / 'figures'
TABLE_DIR = OUT_DIR / 'tables'
SUB_DIR = OUT_DIR / 'submission'

for _d in (CACHE_DIR, FIG_DIR, TABLE_DIR, SUB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CHARS_PATH = DATA_DIR / 'chars_final_with_names.parquet'
FILINGS_PATH = DATA_DIR / '8k_20150101_20260831_identified.parquet'
CHAR_LIST_PATH = DATA_DIR / 'factor_char_list.csv'

SEED = 42
TEST_YEARS = list(range(2021, 2027))

FIRST_TARGET = pd.Timestamp('2015-02-28')
TEST_START = pd.Timestamp('2021-01-31')   # first holding (target) month-end
TEST_END = pd.Timestamp('2026-08-31')     # last holding (target) month-end

MISS_FLAG_CUTOFF = pd.Timestamp('2018-12-31')  # missing-flag selection uses only eom <= this
MISS_FLAG_RATE = 0.20

MIN_PRICE = 5.0
ME_CUTOFF_PCTILE = 0.20

# headline signal: 'pred_auto' = per test year, the candidate with the best mean IC on that
# year's own validation window (walk-forward model choice, no test data)
HEADLINE_SIGNAL = 'pred_auto'
HEADLINE_CANDIDATES = ['pred_ew', 'pred_lgbm_all', 'pred_blend']

# beta model used by the optimizer: 'blume' | 'a15' | 'fusion' | 'kalman'
BETA_MODEL = 'fusion'         # chosen on a validation-only horse race (2019-2020 validation +
# 2016-18 pseudo-history; no test data): realized book beta stays negative under every beta
# model tried (short-leg realized beta exceeds long-leg realized beta even with 'fusion'), so the
# fix is BETA_TARGET below, not a further beta-model change.
# optimizer: robust beta neutrality |beta@w| + BETA_UNC_KAPPA*sqrt(sum w_i^2 var_i) <= tol
BETA_UNC_KAPPA = 0.0  # margin is on the wrong scale because beta_var is dominated by noise in
# the 12-month realized-beta calibration target; kept off.

# net exposure regime (src/portfolio.py optimize_month): 'beta' lets the book carry a small net
# exposure n (|n| <= NET_CAP) whenever that's what beta neutrality actually needs; 'dollar'
# (default) pins both legs to exactly +1/-1 (net==0 always). The validation-only horse race found
# a flexible net did not improve realized beta and drifted the book net long with the signal, so
# 'dollar' is the default; 'beta' stays available and is exercised by its own tests.
NET_MODE = 'dollar'
NET_CAP = 0.25                # |n| <= NET_CAP; competition mandate allows +-0.50, this stays
# well inside it so the book never gets close to the hard limit just to chase beta neutrality.
# quadratic penalty NET_PENALTY*n**2 subtracted from optimize_month's objective -- same units as
# the rest of the objective (signal@w, an L1 turnover term, an L2 weight-concentration term), so
# it only discourages leaving dollar-neutral when the beta constraint actually requires it, rather
# than treating any nonzero net as free. Only used in NET_MODE=='beta'; keeps median |n| ~0.03.
NET_PENALTY = 100.0

# short-side tradability screen: shorts only in the top 60% of the universe by size and
# above the 30th percentile of dollar volume (longs unaffected)
SHORT_SCREEN = True
SHORT_MIN_ME_PCTILE = 0.40
SHORT_MIN_DOLVOL_PCTILE = 0.30

# factor selection inside each training window: validation (2019-2020) showed selection LOWERS
# IC (lgbm_all 0.022 vs 0.032 without; ew 0.014 vs 0.016 without), so it no longer restricts the
# model's features by default.
FEATURE_SELECTION = False
SELECTION_REPORT = True       # still compute and save the per-window factor-selection table as a
# "which factors matter" diagnostic, without restricting the model's features (FEATURE_SELECTION).
SELECTION_FDR_Q = 0.10        # Benjamini-Hochberg on monthly-IC t-stats
SELECTION_LASSO_REPS = 20     # stability-selection subsamples
SELECTION_LASSO_FREQ = 0.6    # keep if selected in >= 60% of subsamples
SELECTION_MIN_PER_GROUP = 3

# FinBERT event geometry (embeddings -> PCA -> k-means event types), fitted on filings dated
# on or before GEOM_FIT_END only
GEOM_PCA_DIMS = 64
GEOM_K = 40
GEOM_FIT_END = pd.Timestamp('2018-12-31')
GEOM_NOVELTY_LOOKBACK_MONTHS = 24

BETA_SHRINK = 0.33  # pre-registered ('blume') beta model, used by src/data.py when BETA_MODEL ==
# 'blume': beta = (1-BETA_SHRINK)*beta_60m + BETA_SHRINK*1.0 (missing beta_60m -> 1.0).

# A15 (beta model fix, docs/SPEC.md section 10; used by src/data.py only when BETA_MODEL == 'a15',
# now an ablation-only path per A16 -- decided after test-period numbers had been seen):
# beta = BETA_INTERCEPT + BETA_SLOPE*b1 + BETA_IVOL*ivp,
# where b1 = BETA_FP_W*clip(betabab_1260d,-1,4) + BETA_FP_C, falling back to a Blume-adjusted
# beta_60m (0.67*beta_60m + 0.33) when betabab_1260d is missing, and to BETA_MISSING when both are
# missing; ivp = within-eom percentile rank of ivol_capm_252d (0.5 if missing). BETA_INTERCEPT/
# BETA_SLOPE/BETA_IVOL are coefficients from an OLS of forward 12-month realized beta on formation
# months 2015-01..2017-12 (pre-2019 only); BETA_MISSING=1.25 is the pre-2019 mean forward beta of
# names missing beta_60m. Nothing here was tuned on validation or test data.
BETA_FP_W = 0.6
BETA_FP_C = 0.4
BETA_MISSING = 1.25
BETA_INTERCEPT = 0.31
BETA_SLOPE = 0.56
BETA_IVOL = 0.25

KEY_ITEMS = ['1.01', '1.02', '2.01', '2.02', '2.05', '2.06', '3.01', '4.01', '4.02', '5.02', '7.01', '8.01']
STATE_VARS = ['mkt_vol12', 'disp']

GROSS = 2.0
MAX_WEIGHT = 0.015
N_CAND = 250 if BETA_MODEL == 'blume' else 350  # A15 raised 250->350 only together with the new
# beta model (validation beta was too short-lopsided at 250; see A15 research log entry); A16
# reverts to the pre-registered 250 under the pre-registered ('blume') beta model.
EMA_ALPHA = 0.5

TURNOVER_PENALTY = 0.5   # placeholder, calibrated later on 2019-2020 validation only
L2_PENALTY = 100.0       # placeholder, calibrated later on 2019-2020 validation only

BETA_TOL = 0.005
SECTOR_TOL = 0.03
SIZE_TOL = 0.05

# ex-ante beta target (src/portfolio.py optimize_month): the constraint is |beta@w -
# BETA_TARGET*sum(w_long)| <= BETA_TOL (with dollar legs sum(w_long)==1, i.e. beta@w in
# [BETA_TARGET-BETA_TOL, BETA_TARGET+BETA_TOL]), not the old |beta@w| <= BETA_TOL. Corrects the
# systematic understatement of the book's market sensitivity: a validation-only horse race found
# realized book beta stays negative (~-0.1 to -0.45) at target 0 because realized short-leg beta
# exceeds long-leg beta even with the improved 'fusion' betas, and the optimizer always pins
# ex-ante beta at the tolerance edge. 0.075 was chosen on 2016-2018 pseudo-history (zero crossing
# ~0.065) and confirmed on 2019-2020 validation; no test data was used.
BETA_TARGET = 0.075

COST_BPS = 10
HURDLE_ANNUAL = 0.04

N_JOBS = int(os.environ.get('ALPHABERT_NJOBS', 6))  # env override so other machines never need to
# edit config.py directly


def load_char_list() -> list[str]:
    df = pd.read_csv(CHAR_LIST_PATH)
    names = df['variable'].tolist()
    assert len(names) == 147
    return names
''')

# ===== src/beta.py =====
beta = _load_module('beta', r'''"""Point-in-time stock-beta estimates + uncertainty (docs/research_log.md, A15/A16 postmortem):
the realized portfolio beta was -0.3 to -0.5 because young stocks missing beta_60m got beta 1.0
(imputed) while their true forward beta is ~1.25-1.7, and the optimizer piled into them on the
short side. This module builds better beta estimates ('fusion', 'kalman') plus a per-stock-month
variance the optimizer can use (config.BETA_UNC_KAPPA), calibrated ONLY on pre-2018 data so the
choice of estimator is never informed by validation/test-period returns. 'blume' and 'a15'
reproduce the pre-existing formulas (previously inline in src/data.py) unchanged, beta_var = 0.

All fitted numbers come from one function, fit_beta_params(raw), which only reads eom <= 2017-12
rows as regressors/features (asserted); its regression TARGET (forward_beta) necessarily looks at
returns through 2018-12 (the end of the last calibration formation month's 12-month forward
window), which is still before any validation/test data (2019+). Results are cached to
CACHE_DIR/'beta_params.json' so the (somewhat expensive) fit only runs once.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression



RAW_ESTIMATORS = ['beta_60m', 'betabab_1260d', 'betadown_252d', 'beta_dimson_21d']

CALIB_START = pd.Timestamp('2015-01-31')
CALIB_END = pd.Timestamp('2017-12-31')     # fit_beta_params reads regressor rows only up to here
HOLDOUT_START = pd.Timestamp('2018-01-31')
HOLDOUT_END = pd.Timestamp('2018-12-31')   # held-out accuracy check window (forward window ends
                                            # 2019-12, still validation-period, never test data)

WINSOR_PCT = 0.01  # 1st/99th percentile winsorization, bounds fit on calibration data only

PRIOR_COLS = ['ivp', 'size_z', 'age_c', 'lev_c']

# bumped whenever fit_beta_params' procedure changes incompatibly with an old cached
# beta_params.json (A16 audit fix: calibration now applies data.universe_mask, see fit_beta_params).
PARAMS_VERSION = 2


# --------------------------------------------------------------------- forward-looking target ---
def forward_beta(raw: pd.DataFrame, mkt: pd.Series) -> pd.Series:
    """Realized beta of each stock over the NEXT 12 months of monthly returns (`ret`) vs the
    value-weighted market return `mkt` (a Series indexed by eom, e.g. data.market_state()['mkt_ret']).
    Computed for every formation month t in `raw` whose full 12-month forward window (t+1..t+12,
    consecutive calendar months) is present with no gaps, for stocks with a non-null `ret` in
    every one of those 12 months (no interpolation -- a stock that stops trading partway through
    the window simply has no forward beta at that formation month). Only ever used as a
    calibration TARGET (never as a feature at prediction time), so looking forward here does not
    break point-in-time-ness of the estimators themselves.
    Returns a Series indexed by (permno, eom) named 'fwd_beta'.
    """
    wide = raw.drop_duplicates(['permno', 'eom']).pivot(index='eom', columns='permno', values='ret')
    eoms = wide.index.sort_values()
    mkt = mkt.reindex(eoms)

    out = {}
    for t in eoms:
        window = pd.date_range(t + pd.offsets.MonthEnd(1), t + pd.offsets.MonthEnd(12), freq='ME')
        if not window.isin(wide.index).all():
            continue
        mkt_sub = mkt.loc[window]
        if mkt_sub.isna().any():
            continue
        sub = wide.loc[window]
        valid_cols = sub.columns[sub.notna().all(axis=0)]
        if len(valid_cols) == 0:
            continue
        sub_v = sub[valid_cols]
        mkt_c = mkt_sub - mkt_sub.mean()
        denom = float((mkt_c ** 2).sum())
        if denom <= 0:
            continue
        sub_c = sub_v - sub_v.mean(axis=0)
        cov = sub_c.mul(mkt_c, axis=0).sum(axis=0)
        beta_t = cov / denom
        for permno, b in beta_t.items():
            out[(permno, t)] = b

    idx = pd.MultiIndex.from_tuples(out.keys(), names=['permno', 'eom'])
    return pd.Series(list(out.values()), index=idx, name='fwd_beta')


# --------------------------------------------------------------------------------- calibration ---
def _ols_1d(x: np.ndarray, y: np.ndarray):
    """Simple OLS y = a + b*x. Returns (a, b, residual variance, n)."""
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    b = np.sum((x - xm) * (y - ym)) / sxx
    a = ym - b * xm
    resid = y - (a + b * x)
    var = float(np.sum(resid ** 2) / max(n - 2, 1))
    return float(a), float(b), var, n


def _prior_features(df: pd.DataFrame, age_bounds=None, lev_bounds=None) -> pd.DataFrame:
    """Cross-sectional prior regressors, computed from raw (unranked) columns of df:
    ivp = within-eom pct rank of ivol_capm_252d (0.5 if missing), size_z = within-eom z-score of
    log(me), age_c = age (months) clipped to age_bounds, lev_c = raw at_be clipped to lev_bounds.
    `age_bounds`/`lev_bounds` are (lo, hi, fill) tuples; when None (fit time), bounds are the
    1st/99th percentile of df itself and fill is the median of the clipped series."""
    ivp = df['ivol_capm_252d'].groupby(df['eom']).rank(pct=True).fillna(0.5)

    log_me = np.log(df['me'])
    size_std = log_me.groupby(df['eom']).transform('std')
    size_z = (log_me - log_me.groupby(df['eom']).transform('mean')) / size_std
    size_z = size_z.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    if age_bounds is None:
        age_lo, age_hi = df['age'].quantile([WINSOR_PCT, 1 - WINSOR_PCT])
        age_c = df['age'].clip(age_lo, age_hi)
        age_fill = float(age_c.median())
        age_bounds = (float(age_lo), float(age_hi), age_fill)
    age_lo, age_hi, age_fill = age_bounds
    age_c = df['age'].clip(age_lo, age_hi).fillna(age_fill)

    if lev_bounds is None:
        lev_lo, lev_hi = df['at_be'].quantile([WINSOR_PCT, 1 - WINSOR_PCT])
        lev_c = df['at_be'].clip(lev_lo, lev_hi)
        lev_fill = float(lev_c.median())
        lev_bounds = (float(lev_lo), float(lev_hi), lev_fill)
    lev_lo, lev_hi, lev_fill = lev_bounds
    lev_c = df['at_be'].clip(lev_lo, lev_hi).fillna(lev_fill)

    gics2 = pd.Series(np.where(df['gics'].isna(), 'NA', df['gics'].astype(str).str.slice(0, 2)),
                       index=df.index)

    return pd.DataFrame({'ivp': ivp, 'size_z': size_z, 'age_c': age_c, 'lev_c': lev_c,
                          'gics2': gics2}, index=df.index), age_bounds, lev_bounds


def fit_beta_params(raw: pd.DataFrame) -> dict:
    """Fit all calibration numbers (per-estimator linear calibration + error variance, the
    cross-sectional prior regression, and the Kalman AR(1) reversion params) on formation months
    2015-01..2017-12 ONLY, and cache to CACHE_DIR/'beta_params.json'.

    `raw` must contain permno/eom/ret/me/gics/age/at_be plus the RAW_ESTIMATORS and
    ivol_capm_252d columns, and must cover at least through 2018-12 (the calibration TARGET,
    forward_beta, looks at returns through the end of the last formation month's 12-month forward
    window -- still before any validation/test data, which starts 2019-01).
    """
    formation = raw.loc[(raw['eom'] >= CALIB_START) & (raw['eom'] <= CALIB_END)].copy()
    assert formation['eom'].min() >= CALIB_START and formation['eom'].max() <= CALIB_END, (
        'fit_beta_params must only use formation (regressor) rows with eom in '
        f'[{CALIB_START.date()}, {CALIB_END.date()}]')


    # A16 audit fix #1: compute_betas (called from data._build) only ever sees universe rows, so
    # the ivp/size_z within-eom ranks it computes at predict time are ranks WITHIN the universe.
    # Restrict the formation (regressor) rows to the same universe here so the fitted prior
    # coefficients are applied on the same rank scale they were fitted on.
    formation = formation.loc[data.universe_mask(formation)].copy()
    mkt = data.market_state()['mkt_ret']
    fwd = forward_beta(raw.loc[raw['eom'] <= HOLDOUT_END], mkt).rename('fwd_beta').reset_index()

    feats, age_bounds, lev_bounds = _prior_features(formation)
    formation = pd.concat([formation, feats], axis=1)
    merged = formation.merge(fwd, on=['permno', 'eom'], how='inner')
    assert len(merged) > 100, 'too few calibration rows with a forward-beta target'

    # 1) per-estimator linear calibration a_j + b_j*x_j -> forward beta, with error variance R_j
    estimators = {}
    for j in RAW_ESTIMATORS:
        sub = merged[[j, 'fwd_beta']].dropna()
        lo, hi = sub[j].quantile([WINSOR_PCT, 1 - WINSOR_PCT])
        xc = sub[j].clip(lo, hi).to_numpy(dtype=float)
        a, b, var, n = _ols_1d(xc, sub['fwd_beta'].to_numpy(dtype=float))
        estimators[j] = {'a': a, 'b': b, 'R': max(var, 1e-6), 'clip_lo': float(lo), 'clip_hi': float(hi), 'n': n}

    # 2) cross-sectional prior: OLS of forward beta on [ivp, size_z, gics2 dummies, age_c, lev_c]
    gics_dummies = pd.get_dummies(merged['gics2'], prefix='gics2', drop_first=True, dtype=float)
    ref_gics2 = sorted(merged['gics2'].unique())[0]
    X = pd.concat([merged[PRIOR_COLS], gics_dummies], axis=1)
    mask = X.notna().all(axis=1) & merged['fwd_beta'].notna()
    X_fit, y_fit = X.loc[mask], merged.loc[mask, 'fwd_beta']
    reg = LinearRegression().fit(X_fit, y_fit)
    resid = y_fit.to_numpy() - reg.predict(X_fit)
    Vp = float(np.sum(resid ** 2) / max(len(y_fit) - X_fit.shape[1] - 1, 1))
    prior = {
        'intercept': float(reg.intercept_),
        'coefs': dict(zip(X_fit.columns, [float(c) for c in reg.coef_])),
        'ref_gics2': ref_gics2, 'Vp': max(Vp, 1e-6),
        'age_bounds': list(age_bounds), 'lev_bounds': list(lev_bounds),
    }
    merged = merged.assign(prior_mean=np.nan)
    merged.loc[mask, 'prior_mean'] = reg.predict(X_fit)

    # 3) Kalman AR(1) reversion toward the prior: phi, q from deviations (fwd_beta - prior_mean)
    # for the same permno 12 CALENDAR MONTHS apart (not consecutive months). forward_beta's
    # 12-month forward windows for two formation months only 1 month apart overlap in 11 of their
    # 12 months, so consecutive-month deviations are far from independent and inflate the fitted
    # phi. Formation months exactly 12 months apart have NON-overlapping forward windows, so:
    #   phi_12 = OLS slope (through the origin) of dev_t on dev_{t-12}
    #   phi    = phi_12 ** (1/12)                      (monthly AR(1) coefficient implied by phi_12)
    #   q      = dev_var * (1 - phi**2)                (so stationary var q/(1-phi**2) matches the
    #                                                    observed deviation variance dev_var)
    dev = merged.dropna(subset=['prior_mean']).sort_values(['permno', 'eom'])[['permno', 'eom', 'fwd_beta', 'prior_mean']]
    dev['dev'] = dev['fwd_beta'] - dev['prior_mean']
    dev_var = float(np.var(dev['dev'].to_numpy(dtype=float), ddof=1)) if len(dev) > 1 else float(prior['Vp'])
    dev['dev_prev12'] = dev.groupby('permno')['dev'].shift(12)
    dev['eom_prev12'] = dev.groupby('permno')['eom'].shift(12)
    pairs12 = dev.loc[dev['eom'] == dev['eom_prev12'] + pd.offsets.MonthEnd(12)].dropna(subset=['dev_prev12'])
    if len(pairs12) >= 30:
        x = pairs12['dev_prev12'].to_numpy(dtype=float)
        y = pairs12['dev'].to_numpy(dtype=float)
        phi_12 = float(np.sum(x * y) / np.sum(x ** 2))
        phi_12 = float(np.clip(phi_12, 0.01, 0.98))  # keep in (0, 1] so the 1/12 root is real
        phi = float(phi_12 ** (1 / 12))
        q = max(dev_var * (1 - phi ** 2), 1e-6)
    else:
        phi, q = 0.8, 0.3 * prior['Vp']  # fallback if too little 12-month-apart data

    params = {
        'estimators': estimators, 'prior': prior,
        'kalman': {'phi': phi, 'q': max(q, 1e-6), 'n_pairs': int(len(pairs12))},
        'calib_start': str(CALIB_START.date()), 'calib_end': str(CALIB_END.date()),
        'version': PARAMS_VERSION, 'universe_masked': True,
    }
    (config.CACHE_DIR / 'beta_params.json').write_text(json.dumps(params, indent=2))
    return params


def _load_or_fit_params() -> dict:
    path = config.CACHE_DIR / 'beta_params.json'
    if path.exists():
        cached = json.loads(path.read_text())
        # A16 audit fix #4: only trust a cached fit if it was produced by this same procedure --
        # same calibration window and the universe-masked formation rows of fix #1. A stale cache
        # (old version, missing field, or a different calib_end) is silently refit instead of
        # reused.
        if (cached.get('version') == PARAMS_VERSION
                and cached.get('calib_end') == str(CALIB_END.date())
                and cached.get('universe_masked') is True):
            return cached

    cols = ['permno', 'eom', 'ret', 'me', 'prc', 'gics', 'age', 'at_be', 'ivol_capm_252d'] + RAW_ESTIMATORS
    raw = data.load_chars(columns=cols)
    raw = raw.loc[raw['eom'] <= HOLDOUT_END]  # only need through 2018-12 for calibration + target
    return fit_beta_params(raw)


# ------------------------------------------------------------------------------- fusion/kalman ---
def _prior_mean_var(df: pd.DataFrame, params: dict):
    pp = params['prior']
    feats, _, _ = _prior_features(df, age_bounds=tuple(pp['age_bounds']), lev_bounds=tuple(pp['lev_bounds']))
    mean = pd.Series(pp['intercept'], index=df.index)
    for col in ('ivp', 'size_z', 'age_c', 'lev_c'):
        mean = mean + pp['coefs'].get(col, 0.0) * feats[col]
    for name, coef in pp['coefs'].items():
        if name.startswith('gics2_'):
            cat = name[len('gics2_'):]
            mean = mean + coef * (feats['gics2'] == cat).astype(float)
    return mean, float(pp['Vp'])


def _calibrated_estimators(df: pd.DataFrame, params: dict) -> dict:
    """{estimator_name: (calibrated_series, R_j)} for estimators present in both df and params."""
    out = {}
    for j in RAW_ESTIMATORS:
        if j not in df.columns or j not in params['estimators']:
            continue
        p = params['estimators'][j]
        x = df[j]
        cal = p['a'] + p['b'] * x.clip(p['clip_lo'], p['clip_hi'])
        out[j] = (cal.where(x.notna()), float(p['R']))
    return out


def _fuse(prior_mean: pd.Series, prior_var: float, cal: dict):
    """Precision-weighted combination of the prior and each available calibrated estimator.
    Missing estimators drop out of the sum entirely (their weight is 0 for that row)."""
    prec = pd.Series(1.0 / prior_var, index=prior_mean.index)
    num = prior_mean / prior_var
    for _j, (cal_j, R_j) in cal.items():
        avail = cal_j.notna().astype(float)
        num = num + (cal_j.fillna(0.0) / R_j) * avail
        prec = prec + avail / R_j
    mean = num / prec
    var = 1.0 / prec
    return mean, var


def _kalman(df: pd.DataFrame, prior_mean: pd.Series, prior_var: float, cal: dict, params: dict):
    phi, q = params['kalman']['phi'], params['kalman']['q']
    est_names = list(cal.keys())

    work = pd.DataFrame({'permno': df['permno'].values, 'eom': df['eom'].values,
                          'prior_mean': prior_mean.values}, index=df.index)
    for j in est_names:
        cal_j, R_j = cal[j]
        work[f'cal_{j}'] = cal_j.values
        work[f'R_{j}'] = R_j
    work = work.sort_values(['permno', 'eom'])

    beta_out = pd.Series(np.nan, index=df.index)
    var_out = pd.Series(np.nan, index=df.index)

    for _permno, g in work.groupby('permno', sort=False):
        x_prev = mu_prev = P_prev = None
        for row in g.itertuples():
            mu_t = row.prior_mean
            if x_prev is None:
                x_pred, P_pred = mu_t, prior_var
            else:
                x_pred = mu_t + phi * (x_prev - mu_prev)
                P_pred = phi ** 2 * P_prev + q

            num, prec_obs = 0.0, 0.0
            for j in est_names:
                c = getattr(row, f'cal_{j}')
                if pd.notna(c):
                    Rj = getattr(row, f'R_{j}')
                    num += c / Rj
                    prec_obs += 1.0 / Rj

            if prec_obs > 0:
                R_comb = 1.0 / prec_obs
                z = num * R_comb
                K = P_pred / (P_pred + R_comb)
                x_t = x_pred + K * (z - x_pred)
                P_t = (1 - K) * P_pred
            else:
                x_t, P_t = x_pred, P_pred

            beta_out.loc[row.Index] = x_t
            var_out.loc[row.Index] = P_t
            x_prev, mu_prev, P_prev = x_t, mu_t, P_t

    return beta_out, var_out


# ------------------------------------------------------------------------------------- public ---
def compute_betas(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Beta + beta_var aligned to df.index, for model in {'blume','a15','fusion','kalman'}.
    df must have raw (unranked) columns: beta_60m, betabab_1260d, betadown_252d, beta_dimson_21d,
    ivol_capm_252d, me, gics, age, at_be, permno, eom.
    """
    if model == 'blume':
        # pre-registered: Blume-shrunk beta_60m, missing beta_60m -> 1.0 (previously inline in
        # src/data.py._build; unchanged here).
        beta = ((1 - config.BETA_SHRINK) * df['beta_60m'] + config.BETA_SHRINK).fillna(1.0)
        beta_var = pd.Series(0.0, index=df.index)
    elif model == 'a15':
        # A15 (ablation-only, docs/SPEC.md section 10): unchanged formula, previously inline in
        # src/data.py._build.
        b1 = ((config.BETA_FP_W * df['betabab_1260d'].clip(-1, 4) + config.BETA_FP_C)
              .fillna(0.67 * df['beta_60m'] + 0.33)
              .fillna(config.BETA_MISSING))
        ivp = df['ivol_capm_252d'].groupby(df['eom']).rank(pct=True).fillna(0.5)
        beta = config.BETA_INTERCEPT + config.BETA_SLOPE * b1 + config.BETA_IVOL * ivp
        beta_var = pd.Series(0.0, index=df.index)
    elif model in ('fusion', 'kalman'):
        params = _load_or_fit_params()
        prior_mean, prior_var = _prior_mean_var(df, params)
        cal = _calibrated_estimators(df, params)
        if model == 'fusion':
            beta, beta_var = _fuse(prior_mean, prior_var, cal)
        else:
            beta, beta_var = _kalman(df, prior_mean, prior_var, cal, params)
    else:
        raise ValueError(f'unknown BETA_MODEL {model!r}')

    return pd.DataFrame({'beta': beta, 'beta_var': beta_var}, index=df.index)


def held_out_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Pre-2019-only accuracy check (never uses validation/test data as a decision input): for
    each model, R^2 and mean bias vs forward beta on formation months 2018-01..2018-12 (forward
    window ends 2019-12), overall and for names missing beta_60m at formation. `raw` must cover
    through 2019-12 (needed for the forward-beta target of Dec-2018 formation).

    LIMITATION (A16 audit fix #2): forward_beta only has a calibration target for a stock at
    formation month t when it has a non-null `ret` in every one of the 12 following months --
    i.e. it survived and kept trading through the full forward window. This report is therefore
    necessarily survivor-conditioned: names that got delisted, went inactive, or otherwise stopped
    reporting returns partway through 2019 are silently excluded from both the numerator and `n`,
    for every model. It is not a claim about accuracy on the full held-out universe, only on the
    subset of it with a usable target.
    """

    mkt = data.market_state()['mkt_ret']
    fwd = forward_beta(raw, mkt).rename('fwd_beta').reset_index()

    # A16 audit fix #2: compute betas on the full held-out UNIVERSE frame first (matching what
    # compute_betas sees at predict time, via data._build) -- ivp/size_z ranks must be taken over
    # this same cohort, not over the smaller survivor-only (has-a-forward-beta-target) subset that
    # merging with `fwd` first would silently restrict them to.
    holdout = raw.loc[(raw['eom'] >= HOLDOUT_START) & (raw['eom'] <= HOLDOUT_END)]
    holdout = holdout.loc[data.universe_mask(holdout)].copy()

    rows = []
    for model in ('blume', 'a15', 'fusion', 'kalman'):
        pred = compute_betas(holdout, model)['beta']
        pred_df = pd.DataFrame({'permno': holdout['permno'].to_numpy(),
                                 'eom': holdout['eom'].to_numpy(),
                                 'beta_60m': holdout['beta_60m'].to_numpy(),
                                 'pred': pred.to_numpy()})
        merged = pred_df.merge(fwd, on=['permno', 'eom'], how='inner')
        for subset_name, mask in [('all', pd.Series(True, index=merged.index)),
                                   ('missing_beta_60m', merged['beta_60m'].isna())]:
            y = merged.loc[mask, 'fwd_beta'].to_numpy()
            p = merged.loc[mask, 'pred'].to_numpy()
            n = len(y)
            if n == 0:
                continue
            resid = y - p
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            rows.append({'model': model, 'subset': subset_name, 'n': n,
                         'r2': r2, 'mean_bias': float(resid.mean())})
    return pd.DataFrame(rows)
''', config=config)

# ===== src/data.py =====
data = _load_module('data', r'''"""Panel construction, universe, ranks, market state, external market data. See docs/SPEC.md section 3."""
import hashlib
import io
import urllib.request

import numpy as np
import pandas as pd





FRED_TB3MS_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS'
FRED_SP500_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500'


def load_chars(columns=None) -> pd.DataFrame:
    df = pd.read_parquet(config.CHARS_PATH, columns=columns)
    for c in ('eom', 'date'):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).astype('datetime64[ns]')
    return df


def pipeline_rf() -> pd.Series:
    """The risk-free rate baked into the data provider's excess-return columns, recovered
    directly from the raw chars file rather than assumed to equal/cancel with rf_m (the T-bill
    rate loaded in load_market()). For any stock-month, ret_exc = ret - rf_pipe, so rf_pipe is
    observable as (ret - ret_exc) -- constant within an eom to ~3e-17; median taken per eom for
    robustness. Indexed by eom (month-end), the same convention ret/ret_exc themselves use --
    NOT lagged like ret_exc_lead1m, so this is the rf baked into the return realized DURING that
    calendar month, i.e. pipeline_rf.loc[h] is the rf baked into holding month h's realized
    ret_exc_lead1m (which was recorded one eom earlier, at h - 1 month-end, as the forward
    return into h)."""
    raw = load_chars(columns=['eom', 'ret', 'ret_exc'])
    rf = (raw['ret'] - raw['ret_exc']).groupby(raw['eom']).median()
    rf.index.name = 'eom'
    rf.name = 'rf_pipe'
    return rf


def universe_mask(raw: pd.DataFrame) -> pd.Series:
    """prc >= MIN_PRICE and me >= the ME_CUTOFF_PCTILE quantile of me among all rows of the same eom.
    Never touches ret_exc_lead1m."""
    thresh = raw.groupby('eom')['me'].transform(lambda s: s.quantile(config.ME_CUTOFF_PCTILE))
    return ((raw['prc'] >= config.MIN_PRICE) & (raw['me'] >= thresh)).fillna(False)


def _rank_within_eom(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """2*(rank-1)/(n-1) - 1 within eom, average ties, over non-null values; NaN (incl. n<=1) -> 0."""
    g = df.groupby('eom')[cols]
    with np.errstate(divide='ignore', invalid='ignore'):
        ranked = g.rank(method='average')
        counts = g.transform('count')
        scaled = 2 * (ranked - 1) / (counts - 1) - 1
    return scaled.fillna(0.0)


def _build(raw: pd.DataFrame) -> pd.DataFrame:
    """Core panel-building logic (universe filter, ranks, miss flags, aux columns) on an
    already-loaded raw frame. Split out from build_panel so it is callable without touching the
    on-disk cache (used by truncation-invariance tests)."""
    chars = config.load_char_list()
    mask = universe_mask(raw)
    df = raw.loc[mask].reset_index(drop=True).copy()
    print(f'universe rows: {len(df)} of {len(raw)} raw rows ({df["eom"].nunique()} months)')

    target_month = df['eom'] + pd.offsets.MonthEnd(1)
    stock_exret = df['ret_exc_lead1m']

    # raw copies needed for aux columns before the same-named 147-char columns get rank-transformed
    prc_raw = df['prc'].copy()
    dolvol_raw = df['dolvol_126d'].copy()

    # miss_ flags: selected on universe rows with eom <= cutoff, applied to all universe rows
    cutoff_rows = df.loc[df['eom'] <= config.MISS_FLAG_CUTOFF, chars]
    miss_rate = cutoff_rows.isna().mean()
    miss_chars = miss_rate[miss_rate > config.MISS_FLAG_RATE].index.tolist()
    print(f'miss_ flags selected ({len(miss_chars)}): {miss_chars}')
    miss_flags = {f'miss_{c}': df[c].isna().astype('int8') for c in miss_chars}

    scaled = _rank_within_eom(df, chars)

    # aux (not features)
    gics2 = pd.Series(np.where(df['gics'].isna(), 'NA', df['gics'].astype(str).str.slice(0, 2)), index=df.index)
    # A16 (2026-09-28, docs/SPEC.md section 10): the beta model (A15) was decided after
    # test-period numbers had been seen, so config.BETA_MODEL selects between the pre-registered
    # design (default), the A15 fix, and the point-in-time-calibrated 'fusion'/'kalman' models
    # (src/beta.py) built to fix the beta_60m-missing -> beta-1.0 imputation bug (docs/research_log.md,
    # A15 entry). `df` still has raw (unranked) chars at this point, which is what compute_betas needs.
    beta_df = compute_betas(df, config.BETA_MODEL)
    beta = beta_df['beta']
    beta_var = beta_df['beta_var']
    log_me = np.log(df['me'])
    size_z = log_me.groupby(df['eom']).transform(lambda s: (s - s.mean()) / s.std())
    # NOTE deviation from literal SPEC wording: 'prc' and 'dolvol_126d' are both feature-char names
    # (ranked above) and requested as raw aux columns under the same name, which collide. We keep
    # the ranked feature under the standard name (needed by models.py) and expose the raw value
    # under '<name>_raw' instead. 'me' has no such collision and stays raw.
    aux = pd.DataFrame({
        **miss_flags,
        'gics2': gics2, 'beta': beta, 'beta_var': beta_var, 'size_z': size_z,
        'prc_raw': prc_raw, 'dolvol_126d_raw': dolvol_raw,
    }, index=df.index)
    lead = pd.DataFrame({'target_month': target_month, 'stock_exret': stock_exret}, index=df.index)

    panel = pd.concat([df[['permno', 'eom']], lead, df[['date']], scaled, aux,
                        df[['me', 'size_grp', 'ticker', 'company_name']]], axis=1)
    return panel


def build_panel() -> pd.DataFrame:
    # A16: the beta model name is baked into the cache filename (panel_blume.parquet /
    # panel_a15.parquet) so a cache built under one config.BETA_MODEL is never silently reused
    # after switching to the other. For 'fusion'/'kalman' (audit fix #4), the filename also carries
    # a short hash of beta_params.json's content, so a refit of those calibration numbers (e.g. the
    # version bump in src/beta.py) can never silently reuse a panel built under the old params --
    # it gets a new cache path instead. Ensure params are current (fit/refit as needed) first.
    if config.BETA_MODEL in ('fusion', 'kalman'):
        beta_mod._load_or_fit_params()
        params_hash = hashlib.md5((config.CACHE_DIR / 'beta_params.json').read_bytes()).hexdigest()[:8]
        cache_path = config.CACHE_DIR / f'panel_{config.BETA_MODEL}_{params_hash}.parquet'
    else:
        cache_path = config.CACHE_DIR / f'panel_{config.BETA_MODEL}.parquet'
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        for c in ('eom', 'target_month', 'date'):
            if c in cached.columns:
                cached[c] = pd.to_datetime(cached[c]).astype('datetime64[ns]')
        return cached

    chars = config.load_char_list()
    id_cols = ['permno', 'eom', 'date', 'prc', 'me', 'gics', 'beta_60m',
               'dolvol_126d', 'size_grp', 'ticker', 'company_name', 'ret_exc_lead1m']
    needed = list(dict.fromkeys(id_cols + chars))
    raw = load_chars(columns=needed)

    panel = _build(raw)
    panel.to_parquet(cache_path)
    return panel


def feature_columns(panel: pd.DataFrame) -> list:
    chars = config.load_char_list()
    chars_present = [c for c in chars if c in panel.columns]
    miss_present = [f'miss_{c}' for c in chars if f'miss_{c}' in panel.columns]
    return chars_present + miss_present


def market_state() -> pd.DataFrame:
    cols = ['permno', 'eom', 'prc', 'me', 'me_lag1', 'ret', 'ivol_capm_21d']
    raw = load_chars(columns=cols)
    mask = universe_mask(raw)

    weight = raw['me_lag1'].fillna(raw['me']).where(raw['ret'].notna())
    w_ret = raw['ret'] * weight
    mkt_ret = w_ret.groupby(raw['eom']).sum() / weight.groupby(raw['eom']).sum()

    uni = raw.loc[mask]
    # clip within-month at the 1st/99th percentile before the cross-sectional std: raw ret has rare
    # outliers up to +3000% that would otherwise dominate an unclipped std.
    clipped_ret = uni.groupby('eom')['ret'].transform(
        lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))
    disp = clipped_ret.groupby(uni['eom']).std()
    ivol = uni.groupby('eom')['ivol_capm_21d'].mean()

    state = pd.DataFrame({'mkt_ret': mkt_ret, 'disp': disp, 'ivol': ivol}).sort_index()
    state['mkt_ret12'] = state['mkt_ret'].rolling(12, min_periods=6).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    state['mkt_vol12'] = state['mkt_ret'].rolling(12, min_periods=6).std()
    state.index.name = 'eom'
    return state


def _current_month_end() -> pd.Timestamp:
    """Month-end of the in-progress (not-yet-closed) calendar month."""
    return pd.Timestamp.today().normalize().to_period('M').to_timestamp('M')


def _download_market_data(tb3ms_path, sp500_path):
    with urllib.request.urlopen(FRED_TB3MS_URL, timeout=30) as r:
        raw_csv = r.read().decode('utf-8')
    tb = pd.read_csv(io.StringIO(raw_csv))
    tb.columns = ['date', 'tb3ms']
    tb['date'] = pd.to_datetime(tb['date'])
    tb['eom'] = tb['date'].dt.to_period('M').dt.to_timestamp('M')
    tb = tb[['eom', 'tb3ms']].dropna()
    tb.to_csv(tb3ms_path, index=False)

    sp500_source = 'sp500tr_total_return'
    try:
        import yfinance as yf
        hist = yf.Ticker('^SP500TR').history(start='2014-11-01', interval='1mo', auto_adjust=False)
        if hist.empty:
            raise RuntimeError('empty yfinance result')
        hist.index = hist.index.tz_localize(None)
        close = hist['Close']
        close.index = close.index.to_period('M').to_timestamp('M')
        ret = close.sort_index().pct_change().dropna()
        sp = ret.rename('sp500_ret').reset_index().rename(columns={'index': 'eom', 'Date': 'eom'})
        sp['sp500_source'] = sp500_source
    except Exception as e:
        print(f'yfinance ^SP500TR failed ({e}); falling back to FRED SP500 price index (price_only).')
        with urllib.request.urlopen(FRED_SP500_URL, timeout=30) as r:
            raw_csv2 = r.read().decode('utf-8')
        px = pd.read_csv(io.StringIO(raw_csv2))
        px.columns = ['date', 'sp500']
        px['date'] = pd.to_datetime(px['date'])
        px = px.dropna()
        px['eom'] = px['date'].dt.to_period('M').dt.to_timestamp('M')
        monthly = px.groupby('eom')['sp500'].last()
        ret = monthly.sort_index().pct_change().dropna()
        sp = ret.rename('sp500_ret').reset_index()
        sp500_source = 'price_only'
        sp['sp500_source'] = sp500_source
    sp.to_csv(sp500_path, index=False)

    caveat = ''
    if sp500_source == 'price_only':
        caveat = ("\nCAVEAT: yfinance ^SP500TR was unavailable; fell back to FRED SP500 (price index, "
                  "no dividends reinvested). sp500_ret therefore understates true total return.\n")
    sources_md = f"""# External market data sources

Downloaded: {pd.Timestamp.today().date()}

- TB3MS (3-Month Treasury Bill Secondary Market Rate, annualized percent, not seasonally adjusted):
  {FRED_TB3MS_URL}
  FRED dates are first-of-month; mapped to that month's month-end for `eom`.
  Units: annual percent (e.g. 4.20 means 4.20%/yr).

- S&P 500 total return (monthly, decimal): yfinance ticker ^SP500TR, monthly interval, Close, pct_change.
  sp500_source = '{sp500_source}'
{caveat}"""
    (config.EXT_DIR / 'SOURCES.md').write_text(sources_md, encoding='utf-8')


def load_market() -> pd.DataFrame:
    tb3ms_path = config.EXT_DIR / 'tb3ms.csv'
    sp500_path = config.EXT_DIR / 'sp500.csv'
    if not (tb3ms_path.exists() and sp500_path.exists()):
        _download_market_data(tb3ms_path, sp500_path)

    tb = pd.read_csv(tb3ms_path, parse_dates=['eom'])
    tb['eom'] = tb['eom'].astype('datetime64[ns]')
    tb = tb.set_index('eom')
    sp = pd.read_csv(sp500_path, parse_dates=['eom'])
    sp['eom'] = sp['eom'].astype('datetime64[ns]')
    sp = sp.set_index('eom')
    df = tb.join(sp, how='inner').sort_index()
    # single place the in-progress current month is dropped: FRED/yfinance can carry a partial
    # bar for it, whether or not an already-cached csv happens to include it.
    df = df.loc[df.index < _current_month_end()]
    df['rf_m'] = df['tb3ms'] / 1200
    df['sp500_exret'] = df['sp500_ret'] - df['rf_m']
    df.index.name = 'eom'
    # the data provider's own risk-free rate (distinct from rf_m above), needed by
    # portfolio.compute_month_return's net-exposure accounting term -- see pipeline_rf().
    df['rf_pipe'] = pipeline_rf().reindex(df.index)

    aug2026 = pd.Timestamp('2026-08-31')
    if aug2026 not in df.index:
        print('WARNING: load_market has no row for 2026-08-31 (TB3MS and/or S&P 500 not yet published).')
    elif df.loc[[aug2026]].isna().any().any():
        print('WARNING: load_market 2026-08-31 row contains NaN values.')
    return df
''', config=config,
    beta_mod=beta, compute_betas=beta.compute_betas)
beta.data = data  # beta.py imports data lazily inside functions (circular at load time)

# ===== src/text.py =====
text = _load_module('text', r'''"""8-K text cleaning, FinBERT scoring (cached), and text features. See docs/SPEC.md section 4.

Event-body design: clean_text() drops the SEC cover page and (where easy) the boilerplate item
title so FinBERT reads the actual event description first, not a press-release exhibit. On GPU
(docs/RUN_FINBERT_DGX.md) compute is not the bottleneck, so MAX_LENGTH defaults to 512 -- BERT's
own positional-embedding limit -- scoring the full cleaned event body per filing rather than just
its opening sentences. This was decided on compute grounds (a device/throughput change), before
any FinBERT score was seen.

Model-side look-ahead guard: ProsusAI/finbert is pinned at FINBERT_REVISION (see docs/SPEC.md A7),
loaded with use_safetensors=False so the pinned .bin weights load rather than a newer safetensors
copy transformers might otherwise prefer from a different ref. Its BERT-base architecture was
pretrained on 2018 corpora (BooksCorpus + Wikipedia), further pretrained on Reuters TRC2
(2008-2010 newswire), and fine-tuned on the Financial PhraseBank (labelled 2014). All three
training sources predate 2021, so scoring 8-Ks filed after that date carries no risk of the frozen
model itself having seen post-filing information -- a guard distinct from the temporal
train/valid/test splits in models.py, which govern the *learned* pipeline.

Cleaning also strips recurring boilerplate that isn't part of the event (measured on a 4% filing
sample: Item 2.02 windows, 33% of the corpus, were >=2-phrase safe-harbor/furnished boilerplate
82.5% of the time; 7.01 55%; 8.01 30%): furnished/"not be deemed filed" disclaimers,
forward-looking-statement safe-harbor sentences, incorporated-by-reference sentences, and
everything from Item 9.01 (exhibit list) or the signature block onward. Company-name/ticker
masking is deliberately narrow to avoid false positives on ordinary words (a bare ticker is masked
only in an explicit exchange context, e.g. "NYSE: TICK"; a short suffix-stripped name variant is
masked case-sensitively and only if it's long/distinctive enough). A filing that cleans to '' is
scored NaN (score_texts), not as an empty string. Consolidated score output is per-setting
(scores_path_for(max_length) -> CACHE_DIR/'finbert_scores_L{max_length}.parquet') so different
max_length/cleaning runs never silently mix.

Embedding pass (`--embed`, embed_texts/embed_finbert/load_embeddings): a second, separate chunked
pass over the same cleaned/truncated text and the same pinned model, used downstream to cluster
filings into event types (PCA -> k-means, docs/SPEC.md section 4) rather than to score tone. Each
filing gets one EMBED_DIM=768 vector: the last hidden state (hidden_states[-1] from the same
classification model, output_hidden_states=True) mean-pooled over the attention mask (padding
excluded), accumulated in fp32. Consolidated output is emb_path_for(max_length) (float16 ndarray)
+ emb_ids_path_for(max_length) (its row-order key), independent of the scores file/cache so a
scoring-only run never needs the embedding pass or vice versa.
"""
import argparse
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq



FINBERT_MODEL = 'ProsusAI/finbert'
FINBERT_REVISION = '4556d13015211d73dccd3fdd39d39232506f3e43'  # pinned commit (HF hub main @ 2026-09-27)
NUM_THREADS = 10
EMBED_DIM = 768  # FinBERT (BERT-base) hidden size
# Decision (2026-09-27, before any GPU result was seen): on the DGX Spark GPU, compute is not
# binding, so MAX_LENGTH is 512 (full event body, BERT's own positional-embedding limit) and
# precision is fp16 -- run `--check` on the target device first to confirm fp16 vs fp32 agreement.
MAX_LENGTH = 512
BATCH_SIZE = 32          # CPU default; GPU default is 256 (see _setup)
GPU_BATCH_SIZE = 256
CHUNK_SIZE = 1000

_ITEM_RE = re.compile(r'I\s*t\s*e\s*m\.?\s{0,3}\d\s{0,2}\.\s{0,2}\d\s{0,2}\d', re.IGNORECASE)
_COVER_FALLBACK_CHARS = 1500

# Some 8-Ks print a "TABLE OF CONTENTS" listing each item's number/title before the real body
# (found scoring a real-filing sample before the DGX run: ~1.7% of filings). When that's present
# before the first _ITEM_RE match, that match lands on the TOC entry rather than the real section
# header, and everything from there is the TOC/SIGNATURES/EXHIBIT INDEX navigation text, not the
# event -- a much worse truncation than missing the item title. The real body re-prints the same
# "Item X.XX <title>" text right before its narrative, so when that exact text repeats later in
# the document, prefer that later occurrence as the body start.
_TOC_RE = re.compile(r'TABLE\s+OF\s+CONTENTS', re.IGNORECASE)
# Item bodies almost always read "Item X.XX <boilerplate title>. On <date>, the Company...".
# Skipping the boilerplate title packs more of the actual event into a short token budget.
# Only applied when the "On " sentence start is found nearby; otherwise the title is kept (safe
# fallback) rather than risk cutting into real content on an unusual filing.
_TITLE_SKIP_RE = re.compile(r'\.\s+(?=On\s)')
_TITLE_SKIP_WINDOW = 300

# Cut everything from Item 9.01 (Financial Statements and Exhibits -- exhibit list, never signal)
# or the signature block onward. Same flexible digit-spacing as _ITEM_RE (real filings occasionally
# have odd whitespace in item numbers, e.g. "Item 5.0 7").
_TAIL_CUT_RE = re.compile(
    r'I\s*t\s*e\s*m\.?\s{0,3}9\s{0,2}\.\s{0,2}0\s{0,2}1'
    r'|SIGNATURES?\b'
    r'|Pursuant\s+to\s+the\s+requirements\s+of\s+the\s+Securities\s+Exchange\s+Act',
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


# Sentence-level boilerplate filters. Measured on a 4% filing sample: Item 2.02 windows (33% of
# the corpus) were >=2-phrase safe-harbor boilerplate 82.5% of the time; 7.01 55%; 8.01 30%. These
# patterns drop the recurring "furnished, not deemed filed" / forward-looking-statements
# disclaimer sentences without touching the actual event sentence.
#
# "incorporated by reference" is handled separately (see _INCORPORATED_BY_REFERENCE_CLAUSE_RE
# below), NOT as a whole-sentence filter: real 8-Ks routinely tack "..., which is hereby
# incorporated by reference." onto the END of the actual event sentence (e.g. "On <date>, <Company>
# issued a press release ... attached as Exhibit 99.1 ..., which is hereby incorporated by
# reference."). A whole-sentence filter on that phrase discarded the entire substantive sentence,
# not just the boilerplate clause -- found scoring a small real-filing sample before the DGX run
# (see docs/research_log.md). Only the trailing clause is stripped; the sentence's real content
# (and its terminating period) survives.
_FURNISHED_RE = re.compile(r'furnished', re.IGNORECASE)
_NOT_DEEMED_RE = re.compile(r'not be deemed', re.IGNORECASE)
_FILED_RE = re.compile(r'\bfiled\b', re.IGNORECASE)
_FORWARD_LOOKING_RES = (
    re.compile(r'forward-looking statements?', re.IGNORECASE),
    re.compile(r'Private Securities Litigation Reform Act', re.IGNORECASE),
    re.compile(r'undue reliance', re.IGNORECASE),
)
# ", which is hereby incorporated by reference[ in its entirety]." -- the common trailing clause.
# Replaced with a single '.' so the sentence still ends cleanly.
_INCORPORATED_BY_REFERENCE_CLAUSE_RE = re.compile(
    r',?\s*(?:which is |and is |is\s+)?hereby\s+incorporated\s+(?:herein\s+)?by\s+reference'
    r'(?:\s+in\s+its\s+entirety)?\.?',
    re.IGNORECASE,
)


def _is_boilerplate_sentence(s: str) -> bool:
    # "shall not be deemed 'filed'"/"is being furnished ... shall not be deemed" -- the Item
    # 2.02/7.01 safe-harbor disclaimer, with or without the word "furnished" (both phrasings occur
    # in practice; requiring "furnished" alone missed the common "shall not be deemed 'filed' for
    # purposes of Section 18..." variant).
    if _NOT_DEEMED_RE.search(s) and (_FURNISHED_RE.search(s) or _FILED_RE.search(s)):
        return True
    if any(p.search(s) for p in _FORWARD_LOOKING_RES):
        return True
    return False


_SUFFIX_RE = re.compile(
    r'(?:[,.]|\s)+(?:incorporated|inc|corporation|corp|company|co|limited|ltd|'
    r'holdings?|group|international|intl|plc|llc|l\.l\.c\.|lp|l\.p\.)\.?\s*$',
    re.IGNORECASE,
)
_PAREN_SUFFIX_RE = re.compile(r'\s*[/(][A-Za-z]{2,6}[/)]\s*$')

# Exchange-context markers a ticker must follow to be masked. A bare ticker (no such marker) is
# left alone: short tickers collide with ordinary words (Target, Box, Post, Team, MA...), so
# unconditional whole-word masking was producing false positives (also masking "Form 8-K" when
# ticker == "FORM"). Group 1 (marker + separator) is kept in the substitution; only the ticker is
# replaced.
_TICKER_CTX_RE = r'((?:NYSE\s+American|NYSE|Nasdaq|NASDAQ|ticker\s+symbol)[:\s]+"?){ticker}(?!\w)'


def _strip_suffix(name: str) -> str:
    prev = None
    name = name.strip()
    while prev != name:
        prev = name
        name = _PAREN_SUFFIX_RE.sub('', name).strip()
        name = _SUFFIX_RE.sub('', name).strip()
    return name


def _name_variants(names: dict) -> tuple:
    """(full_variants, short_variants). full_variants (company_name/content_company_name as
    written) are matched case-insensitively. short_variants (suffix stripped, e.g. "Acme Widgets"
    from "Acme Widgets Corp") are matched case-sensitively, exactly as written (Title Case), and
    only kept when long/distinctive enough (>=2 words or >=6 chars) -- a short single word like
    "Box" or "Target" is too likely to also be an ordinary word to mask unconditionally."""
    full, short = set(), set()
    for key in ('company_name', 'content_company_name'):
        n = names.get(key)
        if isinstance(n, str) and n.strip():
            n = n.strip()
            full.add(n)
            stripped = _strip_suffix(n)
            if len(stripped) >= 3 and (len(stripped.split()) >= 2 or len(stripped) >= 6):
                short.add(stripped)
    return full, short


def clean_text(text, names: dict) -> str:
    """Drop the SEC cover page, cut the Item 9.01/exhibits/signature tail, strip recurring
    safe-harbor/furnished/incorporated-by-reference boilerplate sentences, skip the boilerplate
    item title where easy to find, mask company name/ticker mentions, collapse whitespace.
    Returns '' only when the filing genuinely has no usable text left (score_texts then scores
    it NaN rather than scoring an empty string)."""
    if not text:
        return ''
    text = str(text)
    m = _ITEM_RE.search(text)
    if m and _TOC_RE.search(text, 0, m.start()):
        repeat_at = text.find(m.group(0), m.end())
        if repeat_at != -1:
            later = _ITEM_RE.search(text, repeat_at)
            if later:
                m = later
    if m:
        body = text[m.start():]
    elif len(text) > _COVER_FALLBACK_CHARS:
        body = text[_COVER_FALLBACK_CHARS:]
    else:
        # too short to safely assume the first _COVER_FALLBACK_CHARS chars are a cover page;
        # dropping them would empty the text, so keep it whole instead.
        body = text

    tail = _TAIL_CUT_RE.search(body)
    if tail:
        body = body[:tail.start()]

    if m:
        skip = _TITLE_SKIP_RE.search(body[:_TITLE_SKIP_WINDOW])
        if skip:
            body = body[skip.end():]

    body = _INCORPORATED_BY_REFERENCE_CLAUSE_RE.sub('.', body)
    body = ' '.join(s for s in _SENTENCE_SPLIT_RE.split(body) if not _is_boilerplate_sentence(s))

    # (?<!\w)...(?!\w) rather than \b on both sides: a plain \b fails right after a variant that
    # ends in punctuation (e.g. "Inc.") followed by a space, since both sides are non-word chars.
    full_variants, short_variants = _name_variants(names)
    for variant in sorted(full_variants, key=len, reverse=True):
        pat = re.compile(r'(?<!\w)' + re.escape(variant) + r'(?!\w)', re.IGNORECASE)
        body = pat.sub('the Company', body)
    for variant in sorted(short_variants, key=len, reverse=True):
        pat = re.compile(r'(?<!\w)' + re.escape(variant) + r'(?!\w)')  # case-sensitive, as written
        body = pat.sub('the Company', body)

    ticker = names.get('ticker')
    if isinstance(ticker, str) and len(ticker) >= 2:
        pat = re.compile(_TICKER_CTX_RE.format(ticker=re.escape(ticker)), re.IGNORECASE)
        body = pat.sub(lambda mo: mo.group(1) + 'the Company', body)

    return re.sub(r'\s+', ' ', body).strip()


# ---------------------------------------------------------------- FinBERT scoring
def _setup(device: str, batch_size):
    """Resolve the run's device/batch_size/dtype together: 'auto' -> 'cuda' if available else
    'cpu' ('cpu'/'cuda' pass through); batch_size defaults to GPU_BATCH_SIZE on cuda else
    BATCH_SIZE; dtype is fp16 on GPU (per --check on the target device before a full run), fp32 on
    CPU."""
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if batch_size is None:
        batch_size = GPU_BATCH_SIZE if device == 'cuda' else BATCH_SIZE
    dtype = 'fp16' if device == 'cuda' else 'fp32'
    return device, batch_size, dtype


def chunk_dir_for(max_length: int):
    """Cache dir for a given max_length; different max_lengths never mix."""
    return config.CACHE_DIR / f'finbert_chunks_L{max_length}'


def _load_finbert(device: str = 'cpu', dtype: str = 'fp32'):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(NUM_THREADS)
    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL, revision=FINBERT_REVISION, use_fast=True)
    # use_safetensors=False: pin to the .bin weights at FINBERT_REVISION. Without it, transformers
    # silently prefers a safetensors copy from refs/pr/29 if one exists, which defeats the pin.
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL, revision=FINBERT_REVISION, use_safetensors=False
    )
    model.eval()
    model = model.to(device)
    if device == 'cuda' and dtype == 'fp16':
        model = model.half()
    return tok, model


def score_texts(tok, model, texts, max_length=MAX_LENGTH, batch_size=BATCH_SIZE, device='cpu'):
    """Return array (n, 3) of [pos, neg, neu] probabilities, in id2label order 0/1/2.

    A text that clean_text() fully stripped to '' (all boilerplate/cover page, no usable content)
    scores as NaN rather than being fed to the model as an empty string -- pandas aggregation
    (mean/min/max) then excludes it instead of it silently looking neutral.

    Non-empty batches are formed after sorting by text length (less padding waste), then unsorted
    back. Softmax is always computed in fp32, even when the model runs in fp16, for stable
    probabilities."""
    import torch

    out = np.full((len(texts), 3), np.nan, dtype=np.float32)
    idx_nonempty = [i for i, t in enumerate(texts) if t]
    if not idx_nonempty:
        return out

    sub_texts = [texts[i] for i in idx_nonempty]
    order = sorted(range(len(sub_texts)), key=lambda i: len(sub_texts[i]))
    with torch.inference_mode():
        for i in range(0, len(sub_texts), batch_size):
            idxs = order[i:i + batch_size]
            batch = [sub_texts[j] for j in idxs]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for k, j in enumerate(idxs):
                out[idx_nonempty[j]] = probs[k]
    return out


def embed_texts(tok, model, texts, max_length=MAX_LENGTH, batch_size=BATCH_SIZE, device='cpu'):
    """Return array (n, EMBED_DIM) float32: mean-pooled last hidden state over the attention mask
    (padding tokens excluded), for downstream clustering into event types (docs/SPEC.md section 4).

    Uses the same pinned classification model as score_texts, called with output_hidden_states=True
    so hidden_states[-1] (the final encoder layer, per-token, before the classification head) comes
    from the identical model/revision as the scores -- not a separate load. Pooling is accumulated
    in fp32 even when the model runs in fp16 (hidden states are upcast before the masked sum), for
    the same numerical-stability reason softmax in score_texts is always fp32.

    A text that clean_text() fully stripped to '' scores as NaN (same convention as score_texts),
    not as an embedding of an empty string. Batches are length-sorted then unsorted back, matching
    score_texts."""
    import torch

    out = np.full((len(texts), EMBED_DIM), np.nan, dtype=np.float32)
    idx_nonempty = [i for i, t in enumerate(texts) if t]
    if not idx_nonempty:
        return out

    sub_texts = [texts[i] for i in idx_nonempty]
    order = sorted(range(len(sub_texts)), key=lambda i: len(sub_texts[i]))
    with torch.inference_mode():
        for i in range(0, len(sub_texts), batch_size):
            idxs = order[i:i + batch_size]
            batch = [sub_texts[j] for j in idxs]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc, output_hidden_states=True).hidden_states[-1].float()  # fp32 accumulate
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = pooled.cpu().numpy()
            for k, j in enumerate(idxs):
                out[idx_nonempty[j]] = pooled[k]
    return out


def _read_meta(columns):
    meta = pq.read_table(config.FILINGS_PATH, columns=columns).to_pandas()
    return meta.sort_values('filing_date', kind='mergesort').reset_index(drop=True)


def _texts_for_ids(wanted_ids: set) -> dict:
    """Single streamed pass over document_id+text (~1-2s for this 358MB file on local SSD); far
    cheaper than rescanning per chunk, and 373k raw strings is a modest ~1GB on a 64GB box."""
    texts_by_id = {}
    for batch in pq.ParquetFile(config.FILINGS_PATH).iter_batches(columns=['document_id', 'text'], batch_size=50000):
        b = batch.to_pandas()
        hit = b[b['document_id'].isin(wanted_ids)]
        if len(hit):
            texts_by_id.update(dict(zip(hit['document_id'], hit['text'])))
    return texts_by_id


def _clean_rows(sub: pd.DataFrame, texts_by_id: dict) -> list:
    cleaned = []
    for row in sub.itertuples(index=False):
        raw = texts_by_id.get(row.document_id, '')
        names = {'company_name': row.company_name, 'content_company_name': row.content_company_name, 'ticker': row.ticker}
        cleaned.append(clean_text(raw, names))
    return cleaned


def scores_path_for(max_length: int):
    """Consolidated-output path is settings-specific so runs with different max_length never get
    silently mixed."""
    return config.CACHE_DIR / f'finbert_scores_L{max_length}.parquet'


def score_finbert(max_length=None, device='cpu', batch_size=None):
    """Chunked, resumable FinBERT scoring of all filings -> scores_path_for(max_length).

    max_length/device/batch_size select the run's setting; see MAX_LENGTH's module comment and
    chunk_dir_for() for how different settings get separate chunk caches. The consolidated output
    is written only once every chunk of THIS run's setting exists, and carries max_length/device/
    dtype/revision columns for audit."""
    max_length = MAX_LENGTH if max_length is None else max_length
    device, batch_size, dtype = _setup(device, batch_size)
    chunk_dir = chunk_dir_for(max_length)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_scores_path = scores_path_for(max_length)

    meta = _read_meta(['document_id', 'permno', 'filing_date', 'company_name', 'content_company_name', 'ticker'])
    n = len(meta)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f'score_finbert: {n} filings, {n_chunks} chunks of {CHUNK_SIZE}, '
          f'max_length={max_length}, device={device}, dtype={dtype}, batch_size={batch_size}, '
          f'chunk_dir={chunk_dir}', flush=True)

    remaining_chunks = [c for c in range(n_chunks) if not (chunk_dir / f'part_{c:05d}.parquet').exists()]
    done = n - sum(min((c + 1) * CHUNK_SIZE, n) - c * CHUNK_SIZE for c in remaining_chunks)
    if not remaining_chunks:
        print('score_finbert: all chunks already done')
    else:
        wanted_ids = set(meta['document_id'].iloc[
            [i for c in remaining_chunks for i in range(c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, n))]
        ])
        texts_by_id = _texts_for_ids(wanted_ids)

    tok = model = None
    t0 = time.time()
    for c in remaining_chunks:
        out_path = chunk_dir / f'part_{c:05d}.parquet'
        lo, hi = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, n)
        if tok is None:
            tok, model = _load_finbert(device=device, dtype=dtype)

        sub = meta.iloc[lo:hi]
        cleaned = _clean_rows(sub, texts_by_id)

        probs = score_texts(tok, model, cleaned, max_length=max_length, batch_size=batch_size, device=device)
        res = pd.DataFrame({
            'document_id': sub['document_id'].values,
            'permno': sub['permno'].values,
            'filing_date': sub['filing_date'].values,
            'fb_pos': probs[:, 0],
            'fb_neg': probs[:, 1],
            'fb_neu': probs[:, 2],
            'max_length': max_length,
            'device': device,
            'dtype': dtype,
            'revision': FINBERT_REVISION,
        })
        res.to_parquet(out_path, index=False)
        done += (hi - lo)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta_min = (n - done) / rate / 60 if rate > 0 else float('nan')
        n_empty = int((~np.isfinite(probs[:, 0])).sum())
        print(f'  chunk {c + 1}/{n_chunks} done ({done}/{n}, {done / n:.1%}), '
              f'{rate:.1f} docs/s, ETA {eta_min:.0f} min, {n_empty} empty-text (NaN) this chunk', flush=True)

    parts = sorted(chunk_dir.glob('part_*.parquet'))
    if len(parts) == n_chunks:
        all_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        all_df = all_df.drop_duplicates('document_id').sort_values('filing_date').reset_index(drop=True)
        all_df.to_parquet(out_scores_path, index=False)
        print(f'score_finbert: consolidated {len(all_df)} rows -> {out_scores_path}')
    else:
        print(f'score_finbert: {len(parts)}/{n_chunks} chunks done, not consolidating yet')


def emb_chunk_dir_for(max_length: int):
    """Cache dir for embedding chunks at a given max_length; mirrors chunk_dir_for (scores) but
    kept separate so a scoring-only cache never gets mistaken for an embedding cache."""
    return config.CACHE_DIR / f'finbert_emb_chunks_L{max_length}'


def emb_path_for(max_length: int):
    """Consolidated embeddings path (float16 ndarray, shape (N, EMBED_DIM)); settings-specific
    like scores_path_for."""
    return config.CACHE_DIR / f'finbert_emb_L{max_length}.npy'


def emb_ids_path_for(max_length: int):
    """Row-order key (document_id/permno/filing_date) for emb_path_for's ndarray -- row i of one
    file matches row i of the other."""
    return config.CACHE_DIR / f'finbert_emb_L{max_length}_ids.parquet'


def embed_finbert(max_length=None, device='cpu', batch_size=None):
    """Chunked, resumable FinBERT embedding pass -> emb_path_for(max_length) + emb_ids_path_for.

    Mirrors score_finbert's chunked/resumable pattern (same meta ordering, same chunk_dir-exists
    resume check, same NUM_THREADS/device/dtype setup via _setup), but each chunk writes a .npy
    (float16, mean-pooled embeddings, embed_texts) alongside a small .parquet of
    document_id/permno/filing_date in the same row order, since a single ndarray can't carry ids.
    Consolidation concatenates all chunks, drops any duplicate document_id (defensive, matching
    score_finbert) and sorts by filing_date -- applying the identical row selection to the ndarray
    and the ids frame so they stay row-aligned."""
    max_length = MAX_LENGTH if max_length is None else max_length
    device, batch_size, dtype = _setup(device, batch_size)
    chunk_dir = emb_chunk_dir_for(max_length)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_emb_path = emb_path_for(max_length)
    out_ids_path = emb_ids_path_for(max_length)

    meta = _read_meta(['document_id', 'permno', 'filing_date', 'company_name', 'content_company_name', 'ticker'])
    n = len(meta)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f'embed_finbert: {n} filings, {n_chunks} chunks of {CHUNK_SIZE}, '
          f'max_length={max_length}, device={device}, dtype={dtype}, batch_size={batch_size}, '
          f'chunk_dir={chunk_dir}', flush=True)

    def _chunk_paths(c):
        return chunk_dir / f'part_{c:05d}.npy', chunk_dir / f'part_{c:05d}.parquet'

    def _chunk_done(c):
        emb_p, ids_p = _chunk_paths(c)
        return emb_p.exists() and ids_p.exists()

    remaining_chunks = [c for c in range(n_chunks) if not _chunk_done(c)]
    done = n - sum(min((c + 1) * CHUNK_SIZE, n) - c * CHUNK_SIZE for c in remaining_chunks)
    if not remaining_chunks:
        print('embed_finbert: all chunks already done')
    else:
        wanted_ids = set(meta['document_id'].iloc[
            [i for c in remaining_chunks for i in range(c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, n))]
        ])
        texts_by_id = _texts_for_ids(wanted_ids)

    tok = model = None
    t0 = time.time()
    for c in remaining_chunks:
        emb_out_path, ids_out_path = _chunk_paths(c)
        lo, hi = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, n)
        if tok is None:
            tok, model = _load_finbert(device=device, dtype=dtype)

        sub = meta.iloc[lo:hi]
        cleaned = _clean_rows(sub, texts_by_id)

        emb = embed_texts(tok, model, cleaned, max_length=max_length, batch_size=batch_size, device=device)
        np.save(emb_out_path, emb.astype(np.float16))
        ids_df = pd.DataFrame({
            'document_id': sub['document_id'].values,
            'permno': sub['permno'].values,
            'filing_date': sub['filing_date'].values,
        })
        ids_df.to_parquet(ids_out_path, index=False)
        done += (hi - lo)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta_min = (n - done) / rate / 60 if rate > 0 else float('nan')
        n_empty = int(np.isnan(emb[:, 0]).sum())
        print(f'  chunk {c + 1}/{n_chunks} done ({done}/{n}, {done / n:.1%}), '
              f'{rate:.1f} docs/s, ETA {eta_min:.0f} min, {n_empty} empty-text (NaN) this chunk', flush=True)

    npy_parts = sorted(chunk_dir.glob('part_*.npy'))
    if len(npy_parts) == n_chunks:
        all_emb = np.concatenate([np.load(p) for p in npy_parts], axis=0)
        all_ids = pd.concat(
            [pd.read_parquet(chunk_dir / f'{p.stem}.parquet') for p in npy_parts], ignore_index=True
        )
        all_ids['_row'] = np.arange(len(all_ids))
        all_ids = all_ids.drop_duplicates('document_id').sort_values('filing_date').reset_index(drop=True)
        all_emb = all_emb[all_ids['_row'].to_numpy()]
        all_ids = all_ids.drop(columns='_row')
        np.save(out_emb_path, all_emb)
        all_ids.to_parquet(out_ids_path, index=False)
        print(f'embed_finbert: consolidated {len(all_ids)} rows -> {out_emb_path}, {out_ids_path}')
    else:
        print(f'embed_finbert: {len(npy_parts)}/{n_chunks} chunks done, not consolidating yet')


def load_embeddings(max_length=None):
    """Load the consolidated FinBERT embeddings (embed_finbert's output) -> (ids_df, emb).

    ids_df: document_id/permno/filing_date, filing_date normalized to datetime64[ns]. emb: float32
    ndarray (N, EMBED_DIM) (upcast from the on-disk float16); row i of emb matches row i of
    ids_df. Asserts the two files' row counts match -- a mismatch would silently misassign
    filings to the wrong embedding row."""
    max_length = MAX_LENGTH if max_length is None else max_length
    ids_path, emb_path = emb_ids_path_for(max_length), emb_path_for(max_length)
    ids_df = pd.read_parquet(ids_path).reset_index(drop=True)
    emb = np.load(emb_path).astype(np.float32)
    assert len(ids_df) == emb.shape[0], (
        f'{ids_path.name} has {len(ids_df)} rows but {emb_path.name} has {emb.shape[0]} rows'
    )
    ids_df['filing_date'] = pd.to_datetime(ids_df['filing_date']).astype('datetime64[ns]')
    return ids_df, emb


def check_precision(n: int = 500, device: str = 'cpu', max_length=None, batch_size=None):
    """Precision check: score n docs in fp32 and in this device's fast dtype, print corr(pos-neg)
    and max abs diff. Run this on the DGX before a full --score (expect corr > 0.999 for fp16)."""
    max_length = MAX_LENGTH if max_length is None else max_length
    device, batch_size, fast_dtype = _setup(device, batch_size)

    meta = _read_meta(['document_id', 'permno', 'filing_date', 'company_name', 'content_company_name', 'ticker'])
    sub = meta.iloc[:n]
    texts_by_id = _texts_for_ids(set(sub['document_id']))
    cleaned = _clean_rows(sub, texts_by_id)
    print(f'check_precision: n={len(cleaned)}, device={device}, max_length={max_length}, '
          f'batch_size={batch_size}, fast_dtype={fast_dtype}', flush=True)

    tok, model_fp32 = _load_finbert(device=device, dtype='fp32')
    probs_fp32 = score_texts(tok, model_fp32, cleaned, max_length=max_length, batch_size=batch_size, device=device)

    if fast_dtype == 'fp32':
        probs_fast = probs_fp32  # no faster dtype on this device; check path is a no-op identity
    else:
        _, model_fast = _load_finbert(device=device, dtype=fast_dtype)
        probs_fast = score_texts(tok, model_fast, cleaned, max_length=max_length, batch_size=batch_size, device=device)

    # exclude docs that clean_text() stripped to '' (scored NaN by score_texts) from the
    # precision comparison -- there's nothing to compare for them.
    mask = np.isfinite(probs_fp32).all(axis=1) & np.isfinite(probs_fast).all(axis=1)
    n_skipped = int((~mask).sum())
    tone_fp32 = probs_fp32[mask, 0] - probs_fp32[mask, 1]
    tone_fast = probs_fast[mask, 0] - probs_fast[mask, 1]
    corr = float(np.corrcoef(tone_fp32, tone_fast)[0, 1]) if len(tone_fp32) > 1 else float('nan')
    max_abs_diff = float(np.abs(probs_fp32[mask] - probs_fast[mask]).max()) if mask.any() else float('nan')
    print(f'check_precision: corr(pos-neg) fp32 vs {fast_dtype} = {corr:.6f}, max abs diff = {max_abs_diff:.6f} '
          f'({n_skipped} of {len(cleaned)} docs had empty cleaned text, excluded)')
    return corr, max_abs_diff


# ---------------------------------------------------------------- text features
# NOTE (survivorship leak): this 8-K archive is retrospectively assembled, and same-month filing
# coverage strongly encodes hindsight about how close a stock-month is to its permno leaving the
# panel -- same-month filing coverage is only ~1% in the stock-month a permno is last observed,
# rising through a clear gradient by months-to-exit to ~60% for permnos still present at the
# panel's end (see filing_coverage_by_exit below). `has_filing` must therefore NOT be used as a
# model feature (it would leak forward survival), so it is excluded from TEXT_FEATURES. It is
# still produced by build_text_features/add_text_features as an aux column so models.py can
# restrict the text specialist to filer rows and zero out the text forecast for non-filers.
TEXT_FEATURES = (
    ['n_filings']
    + [f'item_{c.replace(".", "_")}' for c in config.KEY_ITEMS]
    + ['tone_mean', 'tone_min', 'fb_neg_max']
)


def build_text_features() -> pd.DataFrame:
    """Per (permno, eom=month-end of filing_date): filing counts, item counts, and FinBERT tone.

    Not cached to parquet -- this rebuilds in seconds from the (cached) FinBERT scores, and an
    on-disk text_features.parquet risked going stale silently whenever the scores file changed
    (e.g. a re-run with different cleaning). Uses MAX_LENGTH's scores file only (scores_path_for);
    a run at a different max_length must be renamed/promoted to that file before this picks it up."""
    filings = pq.read_table(
        config.FILINGS_PATH, columns=['document_id', 'permno', 'filing_date', 'items']
    ).to_pandas()
    filings['filing_date'] = pd.to_datetime(filings['filing_date']).astype('datetime64[ns]')
    filings['eom'] = filings['filing_date'] + pd.offsets.MonthEnd(0)

    feat = filings.groupby(['permno', 'eom']).size().rename('n_filings').reset_index()
    feat['has_filing'] = 1

    item_cols = [f'item_{c.replace(".", "_")}' for c in config.KEY_ITEMS]
    exploded = filings[['permno', 'eom', 'items']].explode('items')
    exploded = exploded[exploded['items'].isin(config.KEY_ITEMS)]
    counts = exploded.groupby(['permno', 'eom', 'items']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=config.KEY_ITEMS, fill_value=0)
    counts.columns = item_cols
    feat = feat.merge(counts.reset_index(), on=['permno', 'eom'], how='left')
    for col in item_cols:
        feat[col] = feat[col].fillna(0).astype('int64')

    scores_path = scores_path_for(MAX_LENGTH)
    if scores_path.exists():
        scores = pd.read_parquet(scores_path)
        missing = set(filings['document_id']) - set(scores['document_id'])
        assert not missing, (
            f'{scores_path.name} is missing {len(missing)} document_id(s) present in {config.FILINGS_PATH.name}; '
            'run score_finbert() to completion at this max_length before building text features'
        )
        assert (scores['max_length'] == MAX_LENGTH).all(), (
            f'{scores_path.name} contains rows scored at a max_length other than {MAX_LENGTH}'
        )
        assert (scores['revision'] == FINBERT_REVISION).all(), (
            f'{scores_path.name} contains rows scored with a FinBERT revision other than {FINBERT_REVISION}'
        )
        scores['filing_date'] = pd.to_datetime(scores['filing_date']).astype('datetime64[ns]')
        scores['eom'] = scores['filing_date'] + pd.offsets.MonthEnd(0)
        scores['tone'] = scores['fb_pos'] - scores['fb_neg']
        tone = scores.groupby(['permno', 'eom']).agg(
            tone_mean=('tone', 'mean'),
            tone_min=('tone', 'min'),
            fb_neg_max=('fb_neg', 'max'),
        ).reset_index()
        feat = feat.merge(tone, on=['permno', 'eom'], how='left')
        # feat rows here all have >=1 filing (feat comes from grouping filings itself). Given the
        # coverage assert above, a NaN after this left-merge can only happen when every filing in
        # the (permno, eom) group had clean_text() return '' (fully stripped boilerplate/cover
        # page) -- those docs score NaN in score_texts, so a group made entirely of such docs
        # has an all-NaN tone_mean/tone_min/fb_neg_max. fillna(0.0) treats that rare degenerate
        # case as neutral rather than propagating NaN into the panel.
        for c in ('tone_mean', 'tone_min', 'fb_neg_max'):
            feat[c] = feat[c].fillna(0.0)
    else:
        print(f'build_text_features: WARNING {scores_path.name} missing -> tone columns omitted')

    feat = feat.sort_values(['permno', 'eom']).reset_index(drop=True)
    return feat


def add_text_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Left-join text features onto panel (permno, eom); zero-fill stock-months with no filing.

    Adds `has_filing` alongside TEXT_FEATURES as a state flag for models.py to gate the text
    specialist on (it is not itself a model feature -- see the survivorship-leak note above
    TEXT_FEATURES). build_text_features() is not cached to parquet (see its docstring), so this
    rebuilds it every call -- a few seconds, and guarantees it always matches the current scores."""
    feat = build_text_features()

    # has_filing is always present in feat (built from the filings themselves); only the
    # FinBERT-derived TEXT_FEATURES columns are conditional on the scores file existing.
    join_cols = [c for c in TEXT_FEATURES if c in feat.columns] + ['has_filing']
    out = panel.merge(feat[['permno', 'eom'] + join_cols], on=['permno', 'eom'], how='left')
    for c in join_cols:
        out[c] = out[c].fillna(0)
    out['has_filing'] = out['has_filing'].astype('int8')
    return out


def filing_coverage_by_exit() -> pd.DataFrame:
    """Diagnostic for the survivorship-leak note above: same-month 8-K filing coverage for
    universe stock-months, bucketed by how close the stock-month is to its permno's last
    appearance anywhere in the raw characteristics data (data.load_chars(), not just the universe
    panel). Confirms why `has_filing` is excluded from TEXT_FEATURES: coverage is lowest right at
    exit and rises with distance from it, reaching roughly its base rate for permnos still present
    at the panel's end.

    The raw (unfiltered) characteristics data, not the universe-filtered panel, decides "exit" vs
    "survivor": a permno can drop out of the *universe* (price/size threshold) while still trading
    and filing normally, which would otherwise get misclassified as an "exit" and dilute the real
    effect this diagnostic measures -- a permno's true last appearance in the raw data (delisting,
    acquisition, etc.) is what should predict near-zero filing coverage, not a temporary
    universe-membership drop-out. Coverage is still reported only over universe stock-months (the
    rows TEXT_FEATURES/has_filing actually apply to).

    A permno's last raw eom counts as a true "exit" only when it falls 12+ months before the raw
    data's last eom -- a permno last seen more recently than that is not distinguishable from one
    that would have reappeared just past the data's edge (right-censoring), so its universe rows
    are excluded entirely (neither an exit bucket nor "survivor"). A permno whose last raw eom IS
    the data's last eom is an unambiguous survivor (directly observed still present); all of its
    universe stock-months (at any distance from the data's end) go in the survivor bucket.

    Returns a DataFrame with columns bucket, n, coverage, in bucket order: 'last panel row',
    '1-2 months before exit', '3-5 months before exit', '6-11 months before exit',
    '12+ months before exit', 'survivor to panel end'."""


    raw_last_eom = data.load_chars(columns=['permno', 'eom']).groupby('permno')['eom'].max()
    panel_end = raw_last_eom.max()

    panel = data.build_panel()[['permno', 'eom']].drop_duplicates().sort_values(['permno', 'eom']).reset_index(drop=True)
    panel = panel.merge(raw_last_eom.rename('last_eom'), on='permno', how='left')

    months_from_end = (panel_end.year - panel['last_eom'].dt.year) * 12 + (panel_end.month - panel['last_eom'].dt.month)
    is_exit = (months_from_end >= 12).to_numpy()
    is_survivor = (months_from_end == 0).to_numpy()
    # months_from_end in 1..11: ambiguous (could be a true exit not yet confirmed by 12 clear
    # months, or a permno that would have reappeared past the data's edge) -- excluded below.

    months_before_exit = (
        (panel['last_eom'].dt.year - panel['eom'].dt.year) * 12 + (panel['last_eom'].dt.month - panel['eom'].dt.month)
    ).to_numpy()

    feat = build_text_features()[['permno', 'eom', 'n_filings']]
    key = panel.merge(feat, on=['permno', 'eom'], how='left')
    has_filing = (key['n_filings'].fillna(0) > 0).to_numpy()

    buckets = [
        ('last panel row', is_exit & (months_before_exit == 0)),
        ('1-2 months before exit', is_exit & (months_before_exit >= 1) & (months_before_exit <= 2)),
        ('3-5 months before exit', is_exit & (months_before_exit >= 3) & (months_before_exit <= 5)),
        ('6-11 months before exit', is_exit & (months_before_exit >= 6) & (months_before_exit <= 11)),
        ('12+ months before exit', is_exit & (months_before_exit >= 12)),
        ('survivor to panel end', is_survivor),
    ]
    rows = [
        {'bucket': label, 'n': int(mask.sum()), 'coverage': float(has_filing[mask].mean()) if mask.any() else float('nan')}
        for label, mask in buckets
    ]
    out = pd.DataFrame(rows)

    n_censored = int((~is_exit & ~is_survivor).sum())
    print(f'filing_coverage_by_exit (excluded as right-censored: {n_censored}):\n{out.to_string(index=False)}')
    return out


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score', action='store_true', help='run the full chunked FinBERT scoring pass')
    parser.add_argument('--embed', action='store_true',
                         help='run the full chunked FinBERT embedding pass (mean-pooled 768-d vectors)')
    parser.add_argument('--check', type=int, nargs='?', const=500, default=None, metavar='N',
                         help='precision check: score N docs (default 500) in fp32 vs the fast dtype, then exit')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-length', type=int, default=None, help=f'default {MAX_LENGTH}')
    parser.add_argument('--batch-size', type=int, default=None, help='default 32 (CPU) / 256 (GPU)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args()
    if args.check is not None:
        check_precision(n=args.check, device=args.device, max_length=args.max_length, batch_size=args.batch_size)
    elif args.score or args.embed:
        # Separate passes (each its own chunked/resumable model pass) rather than one combined
        # forward pass -- keeps score_finbert/embed_finbert independently resumable and simple;
        # --score --embed in one invocation just runs both in turn.
        if args.score:
            score_finbert(max_length=args.max_length, device=args.device, batch_size=args.batch_size)
        if args.embed:
            embed_finbert(max_length=args.max_length, device=args.device, batch_size=args.batch_size)
    else:
        print('usage: python -m src.text --score [--embed] [--device {auto,cpu,cuda}] [--max-length N] [--batch-size N]')
        print('       python -m src.text --embed [--device {auto,cpu,cuda}] [--max-length N] [--batch-size N]')
        print('       python -m src.text --check [N] [--device {auto,cpu,cuda}]')
''', config=config, data=data)

# ===== src/geometry.py =====
geometry = _load_module('geometry', r'''"""FinBERT embedding "event geometry" features: how unusual a filing's disclosure is relative to
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
''', config=config, text=text)

# ===== src/selection.py =====
selection = _load_module('selection', r'''"""Per-training-window factor selection (SPEC: "which stock-level factors are most predictive").

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
        model = LassoCV(cv=3, alphas=20, max_iter=2000, random_state=config.SEED)
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
''', config=config)

# ===== src/models.py =====
models = _load_module('models', r'''"""Schedule, baselines, specialists, gate, OOS R2 / IC (SPEC section 5).

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
''', config=config,
    geometry=geometry, selection=selection,
    feature_columns=data.feature_columns, TEXT_FEATURES=text.TEXT_FEATURES)

# ===== src/portfolio.py =====
portfolio = _load_module('portfolio', r'''"""Signal smoothing, optimizer, backtest accounting, submission files. See docs/SPEC.md section 6."""
import numpy as np
import pandas as pd
import cvxpy as cp



SOLVERS = ['CLARABEL', 'SCS']
DUST = 1e-5  # |w| below this is solver noise, not a real position (SPEC section 6)
# (sector_tol_mult, size_tol_mult, beta_tol_mult, label): base, then sector x2, size x2, beta x2
RELAX_STEPS = [
    (1, 1, 1, ''),
    (2, 1, 1, 'sector x2'),
    (2, 2, 1, 'sector x2,size x2'),
    (2, 2, 2, 'sector x2,size x2,beta x2'),
]


# ---------------------------------------------------------------- smoothing
def smooth(df, col):
    """z-score `col` within eom, then per-permno EMA over formation months (past-only,
    reset after a skipped month). Returns a Series aligned to df.index."""
    d = df[['permno', 'eom', col]].copy()
    grp = d.groupby('eom')[col]
    mu = grp.transform('mean')
    sd = grp.transform('std', ddof=0)
    d['_z'] = ((d[col] - mu) / sd).where(sd > 0, 0.0).fillna(0.0)
    d = d.sort_values(['permno', 'eom'])
    mkey = (d['eom'].dt.year * 12 + d['eom'].dt.month).to_numpy()
    permno = d['permno'].to_numpy()
    z = d['_z'].to_numpy()
    out = np.empty(len(d))
    prev_permno, prev_mkey, s_prev = None, None, 0.0
    for i in range(len(d)):
        if permno[i] != prev_permno or mkey[i] != prev_mkey + 1:
            s = z[i]
        else:
            s = config.EMA_ALPHA * z[i] + (1 - config.EMA_ALPHA) * s_prev
        out[i] = s
        prev_permno, prev_mkey, s_prev = permno[i], mkey[i], s
    return pd.Series(out, index=d.index).reindex(df.index)


# ---------------------------------------------------------------- optimizer
def _solve(prob):
    """Try each solver in turn; only an 'optimal' status counts (never
    'optimal_inaccurate' or anything else)."""
    for solver in SOLVERS:
        try:
            prob.solve(solver=getattr(cp, solver))
        except Exception:
            continue
        if prob.status == 'optimal':
            return True
    return False


def _dust_and_rescale(wv, is_long, is_short, long_target=1.0, short_target=-1.0):
    """Zero-out dust (|w| < DUST), then rescale each leg back to exactly its target sum
    (+1/-1 under config.NET_MODE=='dollar'; long_target/short_target let NET_MODE=='beta'
    shift each leg's sum by n/2 while keeping gross at 2.0, see `optimize_month`). The rescale
    can nudge a name
    already at/near MAX_WEIGHT slightly past it (the solver only enforces the cap up to its
    own tolerance, and rescaling by a ratio > 1 can push it over), so each leg is then
    clipped to <= MAX_WEIGHT, with the clipped excess redistributed proportionally across
    that leg's other names (looped since a redistribution can itself push a different name
    over), leaving the leg sum at exactly its target (up to float epsilon)."""
    wv = wv.copy()
    wv[np.abs(wv) < DUST] = 0.0
    cap = config.MAX_WEIGHT
    for mask, target in ((is_long, long_target), (is_short, short_target)):
        idx = np.where(mask)[0]
        if not idx.size or wv[idx].sum() == 0:
            continue
        wv[idx] = wv[idx] * (target / wv[idx].sum())
        sign = np.sign(target)
        m = wv[idx] * sign  # magnitudes, all >= 0, sum == 1
        for _ in range(10):
            over = m > cap
            if not over.any():
                break
            excess = float((m[over] - cap).sum())
            m[over] = cap
            under = ~over
            room = m[under].sum()
            if room <= 0:
                break
            m[under] += excess * (m[under] / room)
        wv[idx] = sign * m
    return wv


def _tol_groups(sector, filer):
    """Exposure groups sharing SECTOR_TOL and the sector relaxation step: each GICS2
    sector, plus the has_filing==1 group -- the filer-net-neutral constraint (A10: 8-K
    coverage encodes future survival, so filers must not be pushed into the signal tails
    as a group; has_filing is mandatory, see A10/`optimize_month`)."""
    groups = {f'sector {s}': sector == s for s in np.unique(sector)}
    groups['filer'] = filer == 1
    return groups


def _group_shares(m):
    """Each GICS2 sector's share of this month's WHOLE universe (`m`, already dropna'd/
    deduped, indexed by permno -- before candidate selection) by count, plus the has_filing
    group's share, keyed the same way as `_tol_groups`'s group names. Used to build the
    NET_MODE=='beta' relative sector/filer constraint |sum_{i in g} w_i - n*s_g| <= tol (see
    `optimize_month`): a group's allowed net exposure scales with the book's own net exposure
    n in proportion to how much of the universe that group represents, rather than being
    pinned at zero regardless of n."""
    total = len(m)
    shares = {f'sector {s}': float(c) / total for s, c in m['gics2'].value_counts().items()}
    shares['filer'] = float((m['has_filing'] == 1).sum()) / total
    return shares


def _check_constraints(w, is_long, is_short, beta, size_z, groups, shares, cap,
                        sector_tol, size_tol, beta_tol, tol=1e-5, check_n_names=True,
                        long_target=1.0, short_target=-1.0, beta_var=None, kappa=0.0,
                        net_cap=None):
    """One assert block for every optimize_month constraint, at `tol`. `sector_tol`/
    `size_tol`/`beta_tol` are the EFFECTIVE tolerances actually used for the solve that
    produced `w` (a RELAX_STEPS ladder step may have widened them beyond the config base
    values) -- checking against the fixed config constants regardless of which step solved
    would fire spuriously whenever solve_ladder needed to relax. Legs summing to their
    targets (+1/-1 under NET_MODE=='dollar'; long_target/short_target != 1/-1 only under
    NET_MODE=='beta', see `optimize_month`), the MAX_WEIGHT cap, and the 100..500 name count
    are competition rules at fixed values and are never relaxed, so they stay checked against
    fixed constants (or the caller-supplied leg targets, which are themselves fixed for a
    given month before the ladder runs). `check_n_names=False` (used by optimize_month's
    dust/rescale checks that run BEFORE the A15 cardinality guard) skips the upper-bound side
    of the 100..500 name count, which the guard is responsible for fixing; the final check
    after the guard always uses the default check_n_names=True. `beta_var`/`kappa`
    (BETA_UNC_KAPPA robust beta neutrality): when given, the beta check adds
    kappa*norm2(sqrt(beta_var)*w) on top of the (target-adjusted) beta exposure -- the same
    second-order-cone term added to the optimizer's own constraint, so a feasible solve always
    passes this check.

    `net_cap`: when given, asserts |n| <= net_cap (config.NET_CAP under NET_MODE=='beta';
    harmless under 'dollar', where n is 0 by construction). `n` itself is derived from `w`
    (long leg sum + short leg sum), never passed in separately -- it is exactly what the
    dust/rescale step realized. The group check is always the RELATIVE form
    |sum_{i in g} w_i - n*s_g| <= sector_tol (see `_group_shares`): under 'dollar' n is 0 so
    this reduces exactly to the old |sum_{i in g} w_i| <= sector_tol check.

    The beta check is |beta@w - config.BETA_TARGET*sum(w_long)| <= beta_tol (+ robust margin):
    a small positive ex-ante target (chosen on validation, see config.BETA_TARGET) rather than
    the old |beta@w| <= beta_tol, since pinning ex-ante beta at 0 left realized book beta
    negative (short-leg realized beta exceeds long-leg realized beta). Under 'dollar' legs,
    sum(w_long)==1, so this is exactly beta@w in [BETA_TARGET-beta_tol, BETA_TARGET+beta_tol]."""
    assert abs(w[is_long].sum() - long_target) <= tol, "long leg does not sum to its target"
    assert abs(w[is_short].sum() - short_target) <= tol, "short leg does not sum to its target"
    n_val = float(w[is_long].sum() + w[is_short].sum())
    if net_cap is not None:
        assert abs(n_val) <= net_cap + tol, "net exposure exceeds NET_CAP"
    assert abs(n_val) <= 0.50 + tol, "net exposure exceeds the +-50% competition mandate"
    gross = float(w[is_long].sum() - w[is_short].sum())
    assert abs(gross - 2.0) <= tol, "gross != 2.0"
    assert np.abs(w).max() <= cap + 1e-12, "MAX_WEIGHT breached"
    beta_exposure = abs(beta @ w - config.BETA_TARGET * w[is_long].sum())
    if beta_var is not None and kappa > 0:
        beta_exposure = beta_exposure + kappa * np.sqrt(np.sum(beta_var * w ** 2))
    assert beta_exposure <= beta_tol + tol, "beta exposure breached"
    assert abs(size_z @ w) <= size_tol + tol, "size exposure breached"
    for name, mask in groups.items():
        s_g = shares.get(name, 0.0)
        assert abs(w[mask].sum() - n_val * s_g) <= sector_tol + tol, f"{name} exposure breached"
    if check_n_names:
        n_names = int((w != 0).sum())
        assert 100 <= n_names <= 500, f"n_names={n_names} out of [100,500]"


def optimize_month(m, w_prev, l2=None, tc=None):
    """m: DataFrame(permno, signal, beta, gics2, size_z, has_filing) for one formation
    month, plus optional aux columns `me`, `dolvol_126d_raw` (SHORT_SCREEN) and `beta_var`
    (BETA_UNC_KAPPA robust beta neutrality). has_filing is mandatory (A10): the filer group
    (has_filing==1) gets the same relative exposure constraint as a sector, filer-net-neutral.
    w_prev: Series(permno -> weight) from the prior month (empty for the first month).

    NET_MODE (config), the book's net-exposure regime:
    - 'dollar' (default): both legs are hard-pinned to sum to exactly +1/-1 (net==0 always).
      Sector/filer groups use the plain |sum_{i in g} w_i| <= SECTOR_TOL constraint. A
      validation-only horse race found a flexible net did not improve realized book beta and
      drifted the book net long with the signal, so this is the default; beta neutrality is
      instead reached via config.BETA_TARGET (see the beta constraint below).
    - 'beta': a scalar n (|n| <= config.NET_CAP) lets the legs sum to 1+n/2 and -(1-n/2)
      instead -- gross stays exactly 2.0, net becomes whatever n the solver picks. The
      objective is penalized by -NET_PENALTY*n**2 (same units as the rest of the objective),
      so the book only leaves dollar-neutral when the robust beta constraint
      |beta@w - BETA_TARGET*sum(w_long)| (+ BETA_UNC_KAPPA margin) <= BETA_TOL -- the actual
      neutrality anchor -- needs it to. Sector/filer groups then use the RELATIVE constraint
      |sum_{i in g} w_i - n*s_g| <= SECTOR_TOL, where s_g is that group's share of this
      month's whole universe by count (`_group_shares`): a sector's allowed net exposure
      scales with the book's own net exposure in proportion to how much of the universe it
      is, instead of being pinned at zero regardless of n. Under 'dollar', n is 0 by
      construction, so this reduces exactly to the plain constraint above.

    Beta constraint (config.BETA_TARGET, all NET_MODE values): the optimizer targets
    beta@w == BETA_TARGET*sum(w_long) (+-BETA_TOL), not beta@w == 0 -- a small positive
    ex-ante target chosen on validation to correct a systematic negative realized book beta
    (short-leg realized beta exceeding long-leg realized beta even with the improved 'fusion'
    beta model). Under 'dollar' legs sum(w_long)==1, so this is beta@w in
    [BETA_TARGET-BETA_TOL, BETA_TARGET+BETA_TOL].
    The realized net exposure n is not returned separately -- it is exactly the sum of the
    returned weights (`weights.sum()`), since the dust/rescale step below always rescales
    each leg back to its solved target.

    A14: candidate selection and the objective use the signal DEMEANED WITHIN GICS2 for
    this month (pure within-sector stock selection) -- the raw top/bottom N_CAND by
    cross-sectional signal can be sector-lopsided enough that no weighting satisfies the
    sector-neutrality ladder even at its most relaxed step; under sector-neutral
    constraints the objective is (nearly) invariant to a per-sector shift of the signal,
    so this makes candidate sets sector-balanced at ~no cost to the objective.

    SHORT_SCREEN (config): short candidates are drawn only from names meeting a minimum
    size (`me` >= SHORT_MIN_ME_PCTILE quantile) and liquidity (`dolvol_126d_raw` >=
    SHORT_MIN_DOLVOL_PCTILE quantile) bar, both quantiles taken over this month's whole
    universe (`m`, before candidate selection). Longs are never screened -- shorting an
    illiquid/tiny name is the tradability risk this guards against, not owning one.

    Returns (weights: Series(permno -> weight), relax: str naming which tolerances
    were relaxed to find a feasible solution, '' if none were needed)."""
    l2 = config.L2_PENALTY if l2 is None else l2
    tc = config.TURNOVER_PENALTY if tc is None else tc
    beta_net_mode = config.NET_MODE == 'beta'

    m = m.dropna(subset=['signal']).drop_duplicates('permno').set_index('permno')
    shares = _group_shares(m)
    s = m['signal'] - m.groupby('gics2')['signal'].transform('mean')
    long_cand = s.nlargest(config.N_CAND).index

    if config.SHORT_SCREEN and {'me', 'dolvol_126d_raw'}.issubset(m.columns):
        me_thresh = m['me'].quantile(config.SHORT_MIN_ME_PCTILE)
        dolvol_thresh = m['dolvol_126d_raw'].quantile(config.SHORT_MIN_DOLVOL_PCTILE)
        short_eligible = m.index[(m['me'] >= me_thresh) & (m['dolvol_126d_raw'] >= dolvol_thresh)]
        short_cand = s.reindex(short_eligible).nsmallest(config.N_CAND).index
    else:
        short_cand = s.nsmallest(config.N_CAND).index

    w_prev = w_prev[w_prev != 0]
    names = pd.Index(long_cand.union(short_cand).union(w_prev.index))

    signal = s.reindex(names).fillna(0.0).to_numpy()
    beta = m['beta'].reindex(names).fillna(0.0).to_numpy()
    size_z = m['size_z'].reindex(names).fillna(0.0).to_numpy()
    sector = m['gics2'].reindex(names).fillna('NA').to_numpy()
    filer = m['has_filing'].reindex(names).fillna(0).to_numpy()
    groups = _tol_groups(sector, filer)
    wprev_vec = w_prev.reindex(names).fillna(0.0).to_numpy()

    use_robust_beta = config.BETA_UNC_KAPPA > 0 and 'beta_var' in m.columns
    beta_var = m['beta_var'].reindex(names).fillna(0.0).to_numpy() if use_robust_beta else None
    kappa = config.BETA_UNC_KAPPA if use_robust_beta else 0.0
    beta_var_sqrt = np.sqrt(beta_var) if beta_var is not None else None

    is_long = names.isin(long_cand)
    is_short = names.isin(short_cand)
    is_other = ~(is_long | is_short)
    cap = config.MAX_WEIGHT
    dim = len(names)

    def build_constraints(w, nvar, long_target_expr, short_target_expr, sector_tol, size_tol,
                           beta_tol, fixed_zero):
        # is_long/is_short are never all-False: they come from nlargest/nsmallest(N_CAND)
        # over a dropna'd signal, so as long as m has >=1 row each leg is non-empty (and the
        # n_names assert below requires >=100 anyway). is_other can legitimately be empty
        # (e.g. first month, no candidates left over from w_prev), so that guard stays.
        cons = [w[is_long] >= 0, w[is_long] <= cap, cp.sum(w[is_long]) == long_target_expr,
                w[is_short] <= 0, w[is_short] >= -cap, cp.sum(w[is_short]) == short_target_expr]
        if beta_net_mode:
            cons.append(cp.abs(nvar) <= config.NET_CAP)
        if is_other.any():
            cons.append(w[is_other] == 0)
        # ex-ante beta target (config.BETA_TARGET): the constraint is |beta@w -
        # BETA_TARGET*sum(w_long)| <= beta_tol, not the old |beta@w| <= beta_tol -- with dollar
        # legs sum(w_long)==1, i.e. beta@w in [BETA_TARGET-beta_tol, BETA_TARGET+beta_tol]. A
        # small positive target corrects the systematic negative realized book beta found on
        # validation (see config.BETA_TARGET's comment).
        beta_target_expr = config.BETA_TARGET * long_target_expr
        if beta_var_sqrt is not None:
            # robust beta neutrality (BETA_UNC_KAPPA): a second-order cone term that grows
            # with how much weight sits in high-beta-uncertainty names, so the optimizer
            # can't hide market exposure behind an uncertain beta estimate.
            robust = kappa * cp.norm2(cp.multiply(beta_var_sqrt, w))
            cons.append(cp.abs(beta @ w - beta_target_expr) + robust <= beta_tol)
        else:
            cons.append(cp.abs(beta @ w - beta_target_expr) <= beta_tol)
        cons.append(cp.abs(size_z @ w) <= size_tol)
        for name, mask in groups.items():
            if beta_net_mode:
                cons.append(cp.abs(cp.sum(w[mask]) - nvar * shares.get(name, 0.0)) <= sector_tol)
            else:
                cons.append(cp.abs(cp.sum(w[mask])) <= sector_tol)
        if fixed_zero is not None and fixed_zero.any():
            cons.append(w[fixed_zero] == 0)
        return cons

    def solve_ladder(fixed_zero=None):
        w = cp.Variable(dim)
        nvar = cp.Variable() if beta_net_mode else None
        long_target_expr = 1 + nvar / 2 if beta_net_mode else 1.0
        short_target_expr = -(1 - nvar / 2) if beta_net_mode else -1.0
        net_penalty = config.NET_PENALTY * cp.square(nvar) if beta_net_mode else 0.0
        objective = cp.Maximize(
            signal @ w - tc * cp.norm1(w - wprev_vec) - l2 * cp.sum_squares(w) - net_penalty)
        for sf, zf, bf, label in RELAX_STEPS:
            sector_tol, size_tol, beta_tol = config.SECTOR_TOL * sf, config.SIZE_TOL * zf, config.BETA_TOL * bf
            cons = build_constraints(w, nvar, long_target_expr, short_target_expr,
                                      sector_tol, size_tol, beta_tol, fixed_zero)
            if _solve(cp.Problem(objective, cons)):
                if label:
                    print(f"optimize_month: relaxed tolerances {label}")
                n_val = float(nvar.value) if beta_net_mode else 0.0
                return np.asarray(w.value).ravel(), label, (sector_tol, size_tol, beta_tol), n_val
        raise RuntimeError(
            "optimize_month: solver failed even after relaxation (base tolerances tried: "
            f"sector_tol={config.SECTOR_TOL}, size_tol={config.SIZE_TOL}, beta_tol={config.BETA_TOL}, "
            f"relaxation steps={RELAX_STEPS})")

    wv, relax, tols, n_val = solve_ladder()
    long_target, short_target = 1.0 + n_val / 2.0, -(1.0 - n_val / 2.0)
    dust_mask = np.abs(wv) < DUST
    wv = _dust_and_rescale(wv, is_long, is_short, long_target, short_target)
    try:
        # check_n_names=False: the >500 side of the count is the A15 cardinality guard's job
        # (below), not this dust/rescale check's -- with N_CAND=350 a solve can legitimately
        # come back with up to 2*N_CAND=700 nonzero names before the guard trims it.
        _check_constraints(wv, is_long, is_short, beta, size_z, groups, shares, cap, *tols,
                            check_n_names=False, long_target=long_target, short_target=short_target,
                            beta_var=beta_var, kappa=kappa, net_cap=config.NET_CAP)
    except AssertionError:
        # rescaling the dusted solution broke a constraint: re-solve once with
        # the dusted names fixed at exactly 0, then dust/rescale again. Check against
        # THIS solve's effective tolerances (tols) and THIS solve's own n_val, not the first
        # solve's -- the ladder step (and net exposure) that actually produced the returned
        # weights may differ between the two calls.
        wv, relax2, tols, n_val = solve_ladder(fixed_zero=dust_mask)
        long_target, short_target = 1.0 + n_val / 2.0, -(1.0 - n_val / 2.0)
        relax = ','.join(x for x in (relax, relax2) if x)
        wv = _dust_and_rescale(wv, is_long, is_short, long_target, short_target)
        try:
            # this is the one retry this ladder gets: a constraint failure here is a genuine
            # bug (not just a dusted-solution rescale hiccup), so it must not escape as a bare
            # AssertionError -- backtest() only catches RuntimeError from optimize_month (it
            # needs to keep running other formation months / signals), and an uncaught
            # AssertionError here used to kill the whole run on one bad month.
            _check_constraints(wv, is_long, is_short, beta, size_z, groups, shares, cap, *tols,
                                check_n_names=False, long_target=long_target, short_target=short_target,
                                beta_var=beta_var, kappa=kappa, net_cap=config.NET_CAP)
        except AssertionError as e:
            raise RuntimeError(
                f"optimize_month: constraint check failed after retry: {e}") from e

    # A15 cardinality guard: with N_CAND=350 per side the solve can legitimately return more
    # than the competition's 500-name cap (up to 2*N_CAND=700 nonzero names). If so, keep each
    # leg's 250 largest-|w| names, fix everything else to exactly 0 via the existing fixed_zero
    # mechanism, and re-solve once, then dust/rescale/check as usual.
    n_names = int((wv != 0).sum())
    if n_names > 500:
        keep = np.zeros(dim, dtype=bool)
        for mask in (is_long, is_short):
            idx = np.where(mask & (wv != 0))[0]
            top = idx[np.argsort(-np.abs(wv[idx]))[:250]] if idx.size > 250 else idx
            keep[top] = True
        wv, relax3, tols, n_val = solve_ladder(fixed_zero=~keep)
        long_target, short_target = 1.0 + n_val / 2.0, -(1.0 - n_val / 2.0)
        relax = ','.join(x for x in (relax, relax3, 'cardinality') if x)
        wv_before_dust = wv
        wv = _dust_and_rescale(wv, is_long, is_short, long_target, short_target)
        try:
            # final check, always including the 100..500 name count (default check_n_names=True):
            # this is the guard's post-guard check.
            _check_constraints(wv, is_long, is_short, beta, size_z, groups, shares, cap, *tols,
                                long_target=long_target, short_target=short_target,
                                beta_var=beta_var, kappa=kappa, net_cap=config.NET_CAP)
        except AssertionError:
            # same one-retry pattern as the pre-guard check above: rescaling the dusted
            # post-guard solution broke a constraint, so re-solve once with both the names the
            # guard dropped AND this solve's dusted names fixed at exactly 0, then
            # dust/rescale/check again (unwrapped this time).
            wv, relax4, tols, n_val = solve_ladder(fixed_zero=~keep | (np.abs(wv_before_dust) < DUST))
            long_target, short_target = 1.0 + n_val / 2.0, -(1.0 - n_val / 2.0)
            relax = ','.join(x for x in (relax, relax4) if x)
            wv = _dust_and_rescale(wv, is_long, is_short, long_target, short_target)
            try:
                # same rationale as the pre-guard retry above: this is the one retry the
                # post-guard ladder gets, so a failure here must become a RuntimeError (which
                # backtest() catches and annotates with the formation month), not a bare
                # AssertionError that kills the whole backtest run.
                _check_constraints(wv, is_long, is_short, beta, size_z, groups, shares, cap, *tols,
                                    long_target=long_target, short_target=short_target,
                                    beta_var=beta_var, kappa=kappa, net_cap=config.NET_CAP)
            except AssertionError as e:
                raise RuntimeError(
                    f"optimize_month: constraint check failed after retry: {e}") from e
    else:
        # final check, always including the 100..500 name count (default check_n_names=True): the
        # guard didn't run, so n_names was already <=500 (only the >=100 side and a re-check of
        # the other constraints remain to be confirmed here).
        _check_constraints(wv, is_long, is_short, beta, size_z, groups, shares, cap, *tols,
                            long_target=long_target, short_target=short_target,
                            beta_var=beta_var, kappa=kappa, net_cap=config.NET_CAP)

    weights = pd.Series(wv, index=names)
    weights = weights[weights != 0]
    return weights, relax


# ---------------------------------------------------------------- accounting
def compute_month_return(weights, w_prev, ret, rf_m, rf_pipe, sp500_ret, sp500_exret, beta_map):
    """weights, w_prev, ret, beta_map: Series indexed by permno. ret may contain NaN
    for missing realized returns (treated as 0, tracked in missing_ret_weight).
    turnover/cost are target-to-target: 0.5*sum|w_t - w_prev| vs the PREVIOUS month's
    target weights, not intra-month drifted weights.

    Accounting note (NET_MODE, config): capital is 1; the long leg holds 1+n/2, the short leg
    -(1-n/2) (gross 2, net n = sum(weights)); the remaining -n of capital is cash/margin,
    earning/costing rf_m (the T-bill rate loaded by data.load_market()). `ret` here is
    stock_exret, an EXCESS return already net of the data provider's own risk-free rate
    (rf_pipe, data.pipeline_rf()) -- a short-rate series distinct from rf_m, not assumed to
    equal or cancel with it (Brief p.8). So total_ret = rf_m + sum(w*r_excess) +
    n*(rf_pipe - rf_m): the book earns rf_m on its net cash position, plus the excess return of
    each leg over rf_pipe (ls_ret, unchanged), plus the difference between the two cash-rate
    series on the book's net exposure n. Under NET_MODE=='dollar', n==0 always, so this reduces
    exactly to total_ret = rf_m + ls_ret. `rf_pipe` must be the SAME holding month's rate as
    `ret` (data.pipeline_rf() is indexed the same way ret/ret_exc themselves are, i.e. at the
    holding month's own eom -- see its docstring)."""
    idx = weights.index
    r = ret.reindex(idx)
    missing = r.isna()
    r = r.fillna(0.0)

    long_mask = weights > 0
    short_mask = weights < 0
    long_ret = float((weights[long_mask] * r[long_mask]).sum())
    short_ret = float((weights[short_mask] * r[short_mask]).sum())
    ls_ret = long_ret + short_ret
    n_val = float(weights.sum())
    total_ret = rf_m + ls_ret + n_val * (rf_pipe - rf_m)
    bench_ret = rf_m + config.HURDLE_ANNUAL / 12
    active_ret = total_ret - bench_ret

    all_names = idx.union(w_prev.index)
    dw = weights.reindex(all_names).fillna(0.0) - w_prev.reindex(all_names).fillna(0.0)
    turnover = 0.5 * dw.abs().sum() / config.GROSS
    cost = config.COST_BPS / 1e4 * dw.abs().sum()

    beta_exante = float((weights * beta_map.reindex(idx).fillna(0.0)).sum())

    return dict(
        long_ret=long_ret, short_ret=short_ret, ls_ret=ls_ret, rf_m=rf_m, rf_pipe=rf_pipe,
        total_ret=total_ret, bench_ret=bench_ret, active_ret=active_ret,
        sp500_ret=sp500_ret, sp500_exret=sp500_exret,
        n_long=int(long_mask.sum()), n_short=int(short_mask.sum()),
        gross=float(weights.abs().sum()), net=n_val,
        beta_exante=beta_exante, turnover=float(turnover), cost=float(cost),
        total_ret_net=total_ret - cost, active_ret_net=active_ret - cost,
        missing_ret_weight=float(weights[missing].abs().sum()),
    )


# ---------------------------------------------------------------- backtest
def backtest(signal_df, panel, market, l2=None, tc=None):
    """Loops optimize_month() over every formation month in `signal_df`. `market` must carry
    an 'rf_pipe' column alongside 'rf_m'/'sp500_ret'/'sp500_exret' (data.pipeline_rf(),
    reindexed onto market's eom index) -- the data provider's own risk-free rate, kept
    separate from rf_m rather than assumed to cancel (see compute_month_return). Under
    config.NET_MODE=='beta' each month's book can carry a nonzero net exposure n (see
    `optimize_month`); the realized n is recorded both in `net` (compute_month_return's
    ex-post weights.sum()) and `net_target` (the same value, named separately for callers
    that want to read off "what the optimizer solved for" without reaching into `net`)."""
    months = sorted(signal_df['eom'].unique())
    assert 'has_filing' in panel.columns, \
        "backtest: panel lacks has_filing (mandatory, A10 filer-net-neutral constraint)"
    aux_cols = ['permno', 'eom', 'beta', 'gics2', 'size_z', 'ticker', 'company_name', 'stock_exret', 'has_filing']
    optional_cols = [c for c in ('me', 'dolvol_126d_raw', 'beta_var') if c in panel.columns]
    aux = panel[aux_cols + optional_cols].drop_duplicates(['permno', 'eom'])
    m_cols = ['permno', 'signal', 'beta', 'gics2', 'size_z', 'has_filing'] + optional_cols

    holdings_frames = []
    returns_rows = []
    w_prev = pd.Series(dtype=float)
    for i, mth in enumerate(months):
        sig = signal_df.loc[signal_df['eom'] == mth, ['permno', 'eom', 'signal']]
        m = sig.merge(aux, on=['permno', 'eom'], how='inner')
        assert m[['beta', 'gics2', 'size_z']].notna().all().all(), \
            f"backtest: NaN in beta/gics2/size_z after inner-merge with panel aux for {mth}"

        try:
            w, relax = optimize_month(m[m_cols], w_prev, l2=l2, tc=tc)
        except RuntimeError as e:
            raise RuntimeError(f"backtest: optimize_month infeasible at formation month {mth}: {e}") from e

        mi = m.set_index('permno')
        ret = mi['stock_exret']
        beta_map = mi['beta']

        holding_month = mth + pd.offsets.MonthEnd(1)
        if holding_month not in market.index:
            raise ValueError(f"backtest: holding month {holding_month} missing from market (no silent NaN rf_m)")
        row = market.loc[holding_month]
        rf_m, sp500_ret, sp500_exret = row['rf_m'], row['sp500_ret'], row['sp500_exret']
        rf_pipe = row['rf_pipe']

        rec = compute_month_return(w, w_prev, ret, rf_m, rf_pipe, sp500_ret, sp500_exret, beta_map)
        rec['filer_net'] = float((w * mi['has_filing'].reindex(w.index).fillna(0)).sum())
        rec['month'] = holding_month
        rec['first_month'] = (i == 0)
        rec['relax'] = relax
        rec['net_target'] = float(w.sum())  # the solved net exposure n (== rec['net'])
        returns_rows.append(rec)

        h = w.rename('weight').rename_axis('permno').reset_index()
        h = h.merge(mi[['ticker', 'company_name']].reset_index(), on='permno', how='left')
        h['month'] = holding_month
        h['eom'] = mth
        h['label_source'] = np.where(h['ticker'].notna() & h['company_name'].notna(), 'panel', None)
        holdings_frames.append(h)

        w_prev = w

    holdings = pd.concat(holdings_frames, ignore_index=True)
    holdings = holdings[['month', 'eom', 'permno', 'weight', 'ticker', 'company_name', 'label_source']]
    returns = pd.DataFrame(returns_rows).set_index('month')
    return holdings, returns


def missing_return_sensitivity(holdings, panel, returns, long_fill=-0.30, short_fill=0.30):
    """Adverse re-pricing sensitivity check: adjusts the headline `returns` (from
    backtest(), which 0-fills missing stock_exret) by adding, per holding month, the sum
    of weight*fill over names whose stock_exret was missing (each contributed exactly 0
    to the headline). Only ls_ret/total_ret/active_ret (+net) change; turnover/cost/gross/
    net depend only on weights, so they carry over from `returns` unchanged. Never used to
    fill labels for training -- a side check only."""
    aux = panel[['permno', 'eom', 'stock_exret']].drop_duplicates(['permno', 'eom'])
    h = holdings.merge(aux, on=['permno', 'eom'], how='left')
    missing = h['stock_exret'].isna()
    fill = np.where(h['weight'] > 0, long_fill, short_fill)
    h['adj'] = np.where(missing, h['weight'] * fill, 0.0)

    long_adj = h.loc[missing & (h['weight'] > 0)].groupby('month')['adj'].sum()
    short_adj = h.loc[missing & (h['weight'] < 0)].groupby('month')['adj'].sum()

    out = returns.copy()
    out['long_ret'] = out['long_ret'] + long_adj.reindex(out.index, fill_value=0.0)
    out['short_ret'] = out['short_ret'] + short_adj.reindex(out.index, fill_value=0.0)
    out['ls_ret'] = out['long_ret'] + out['short_ret']
    # same total_ret identity as compute_month_return (net/rf_pipe unaffected by which names
    # had missing returns -- only ls_ret changes here)
    out['total_ret'] = out['rf_m'] + out['ls_ret'] + out['net'] * (out['rf_pipe'] - out['rf_m'])
    out['active_ret'] = out['total_ret'] - out['bench_ret']
    out['total_ret_net'] = out['total_ret'] - out['cost']
    out['active_ret_net'] = out['active_ret'] - out['cost']
    return out


# ---------------------------------------------------------------- calibration
def calibrate(signal_df, panel, market, target_names_per_side=150, target_turnover=0.3):
    """Grid search of L2_PENALTY/TURNOVER_PENALTY, reusing backtest() itself so calibration
    runs the exact backtest path (incl. the has_filing filer-net-neutral constraint via
    panel). Scored on portfolio shape only -- names/side and one-way turnover -- and never
    on stock_exret/returns, even though backtest() computes them as a byproduct of reuse."""
    l2_grid = [100.0, 300.0, 1000.0, 3000.0]
    tc_grid = [0.1, 0.3, 1.0, 3.0]

    rows = []
    for l2 in l2_grid:
        for tc in tc_grid:
            _, returns = backtest(signal_df, panel, market, l2=l2, tc=tc)
            avg_names = float((returns['n_long'] + returns['n_short']).mean() / 2.0)
            avg_to = float(returns.loc[~returns['first_month'], 'turnover'].mean())
            score = abs(avg_names - target_names_per_side) + abs(avg_to - target_turnover) * target_names_per_side
            rows.append(dict(l2=l2, tc=tc, avg_names_per_side=avg_names, avg_turnover=avg_to, score=score))

    table = pd.DataFrame(rows)
    best = table.loc[table['score'].idxmin()]
    return float(best['l2']), float(best['tc']), table


# ---------------------------------------------------------------- labels
def load_label_sources():
    label_panel = pd.read_parquet(config.CHARS_PATH, columns=['permno', 'eom', 'ticker', 'company_name'])
    label_panel['eom'] = pd.to_datetime(label_panel['eom']).astype('datetime64[ns]')
    filing_labels = pd.read_parquet(config.FILINGS_PATH, columns=['permno', 'filing_date', 'ticker', 'company_name'])
    filing_labels['filing_date'] = pd.to_datetime(filing_labels['filing_date']).astype('datetime64[ns]')
    return label_panel, filing_labels


def _asof_fill(missing, source, date_col):
    """For each (permno, eom) in `missing`, find the most recent labelled row of
    `source` with date_col <= eom, per permno. Returns a DataFrame indexed by
    `missing`'s `_idx` with columns ticker, company_name (rows with no match dropped)."""
    src = source.dropna(subset=['ticker', 'company_name']).sort_values(date_col)
    src = src.rename(columns={date_col: 'eom'})[['permno', 'eom', 'ticker', 'company_name']]
    left = missing.sort_values('eom')
    # belt-and-braces: merge_asof requires identical key dtypes, and 'eom' can arrive at ns, us,
    # ms or s resolution depending on which pandas build wrote/read the upstream parquet/csv.
    left = left.assign(eom=left['eom'].astype('datetime64[ns]'))
    src = src.assign(eom=src['eom'].astype('datetime64[ns]'))
    merged = pd.merge_asof(left, src, on='eom', by='permno', direction='backward')
    return merged.dropna(subset=['ticker', 'company_name']).set_index('_idx')


def attach_labels(holdings, label_panel, filing_labels):
    """Fill missing ticker/company_name using the most recent label dated <= eom:
    first the raw panel history, then 8-K filing labels, else 'UNLABELED'."""
    h = holdings.copy()
    h['label_source'] = h['label_source'].where(h['ticker'].notna() & h['company_name'].notna(), None)
    h['_idx'] = np.arange(len(h))

    for source, date_col, tag in ((label_panel, 'eom', 'raw_panel'), (filing_labels, 'filing_date', 'filing')):
        need = h.loc[h['label_source'].isna(), ['_idx', 'permno', 'eom']]
        if not len(need):
            continue
        got = _asof_fill(need, source, date_col)
        if not len(got):
            continue
        sel = h['_idx'].isin(got.index)
        h.loc[sel, 'ticker'] = h.loc[sel, '_idx'].map(got['ticker'])
        h.loc[sel, 'company_name'] = h.loc[sel, '_idx'].map(got['company_name'])
        h.loc[sel, 'label_source'] = tag

    h['label_source'] = h['label_source'].fillna('UNLABELED')
    unlabeled = h['label_source'] == 'UNLABELED'
    h.loc[unlabeled, 'ticker'] = h.loc[unlabeled, 'ticker'].fillna('UNLABELED')
    h.loc[unlabeled, 'company_name'] = h.loc[unlabeled, 'company_name'].fillna('UNLABELED')
    return h.drop(columns=['_idx'])


# ---------------------------------------------------------------- submission
def _round_submission_weights(weight_pct, date, cap):
    """Round WEIGHT (percent) to 6dp so each month's long leg sums to exactly its target
    (100*(1+n/2), +100.000000 when n==0) and short leg to its target (-100*(1-n/2),
    -100.000000 when n==0), capped at `cap`. n is derived directly from that month's own
    (unrounded) weights -- n = sum(weight_pct)/100 -- rather than threaded in separately;
    under config.NET_MODE=='dollar' n is 0 by construction, so this reduces exactly to the
    old +-100 legs. The rounding residual goes on the largest-|weight| name of each leg, only
    if that keeps it within `cap`."""
    out = weight_pct.copy()
    for d, idx in weight_pct.groupby(date).groups.items():
        w = weight_pct.loc[idx]
        n = float(w.sum()) / 100.0
        long_target_pct = 100.0 * (1.0 + n / 2.0)
        short_target_pct = -100.0 * (1.0 - n / 2.0)
        for leg_mask, target in ((w > 0, long_target_pct), (w < 0, short_target_pct)):
            leg = w[leg_mask].round(6)
            if leg.empty:
                continue
            residual = round(target - leg.sum(), 6)
            if residual != 0:
                for i in leg.abs().sort_values(ascending=False).index:
                    if abs(leg[i] + residual) <= cap + 1e-9:
                        leg[i] = round(leg[i] + residual, 6)
                        break
            out.loc[leg.index] = leg
    return out


def write_submission(holdings, returns):
    """total_ret (and active_ret) are the headline, gross of transaction costs; the
    *_net columns are the cost-adjusted companions, included for reference.

    NET_MODE (config): each month's long/short WEIGHT legs sum to 100*(1+n/2) / -100*(1-n/2),
    n derived from that month's own weights (see `_round_submission_weights`) -- the un-hedged
    +-100 legs under NET_MODE=='dollar' (n==0), a small net tilt within NET_CAP (well inside
    the competition's +-50% mandate) under NET_MODE=='beta'. Gross stays exactly 200% either
    way. Each Date must also hold 100..500 names (the competition's cardinality rule),
    asserted directly on the written rows -- belt-and-braces on top of optimize_month's own
    A15 cardinality guard."""
    h = holdings.copy()
    assert h['ticker'].notna().all() and h['company_name'].notna().all(), \
        'write_submission: null TICKER/COMPANY NAME - holdings must go through attach_labels'

    date = h['month'].dt.to_period('M').dt.to_timestamp()  # first day of the holding month
    cap_pct = config.MAX_WEIGHT * 100.0

    weight = _round_submission_weights(h['weight'] * 100.0, date, cap_pct)

    out_h = pd.DataFrame({
        'Date': date.dt.strftime('%Y-%m-%d'),
        'PERMNO': h['permno'],
        'TICKER': h['ticker'],
        'COMPANY NAME': h['company_name'],
        'WEIGHT': weight,
    })
    assert out_h['WEIGHT'].abs().max() <= cap_pct + 1e-9
    for d, g in out_h.groupby('Date')['WEIGHT']:
        n_names = int(g.shape[0])
        assert 100 <= n_names <= 500, f"{d}: n_names={n_names} out of the [100,500] competition range"
        n = float(g.sum()) / 100.0
        assert abs(n) <= 0.50 + 1e-9, f"{d}: net exposure exceeds the +-50% mandate"
        long_target_pct = 100.0 * (1.0 + n / 2.0)
        short_target_pct = -100.0 * (1.0 - n / 2.0)
        assert abs(g[g > 0].sum() - long_target_pct) < 1e-6, f"{d}: long WEIGHT does not sum to its target"
        assert abs(g[g < 0].sum() - short_target_pct) < 1e-6, f"{d}: short WEIGHT does not sum to its target"
        assert abs(long_target_pct + abs(short_target_pct) - 200.0) < 1e-6, f"{d}: gross WEIGHT != 200%"
    out_h.to_csv(config.SUB_DIR / 'holdings.csv', index=False)

    r = returns.reset_index().rename(columns={'index': 'month'})
    r_date = r['month'].dt.to_period('M').dt.to_timestamp()
    out_r = pd.DataFrame({'Date': r_date.dt.strftime('%Y-%m-%d')})
    cols = ['total_ret', 'rf_m', 'bench_ret', 'active_ret', 'ls_ret', 'long_ret', 'short_ret',
            'sp500_ret', 'total_ret_net', 'active_ret_net']
    for c in cols:
        out_r[c] = r[c]
    out_r.to_csv(config.SUB_DIR / 'returns.csv', index=False)

    audit = pd.DataFrame({
        'permno': h['permno'], 'month': date.dt.strftime('%Y-%m-%d'),
        'ticker': h['ticker'], 'company_name': h['company_name'],
        'label_source': h['label_source'],
    })
    audit.to_csv(config.SUB_DIR / 'label_audit.csv', index=False)
''', config=config)

# ===== src/evaluate.py =====
evaluate = _load_module('evaluate', r'''"""Performance statistics, tables and charts for the test-window backtest (SPEC section 7).

Inputs, restricted to config.TEST_START..config.TEST_END (68 months, 2021-01..2026-08):
- `returns`: indexed by holding month `month` (src/portfolio.py), incl. total_ret_net,
  active_ret_net, missing_ret_weight and a bool `first_month` (empty prior book -> turnover
  there is a real number, not NaN, but excluded from every turnover figure via `_turnover()`).
  Always carries `relax` (str, '' if optimize_month needed no tolerance relaxation that month)
  and `filer_net` (A10 filer-net exposure diagnostic), read by exposure_table().
- `holdings`: month, eom, permno, weight, ticker, company_name, label_source.
- `panel`: permno, eom, stock_exret, raw `me`, `size_grp`, `dolvol_126d_raw` (the panel's own
  `dolvol_126d` is within-eom RANKED, unusable for short_book_table's dollar checks).
- `gate_coefs` (optional): one row per (test_year, specialist, term, coef), term 'base' (b_k)
  or a config.STATE_VARS name (c_kj), plus per-test_year `state_mean_<var>`/`state_std_<var>`.
  Effective weight w_k(t) = base_k(Y) + sum_j c_kj(Y) * standardized(s_j(t); Y), Y = test_year
  of t's target month.
- `state` (optional, paired with gate_coefs): indexed by eom, raw config.STATE_VARS columns.
- `ablations` (optional): dict {name: returns-shaped DataFrame}, each restricted/asserted the
  same way as `returns`.
- `sensitivity_returns` (optional, run_evaluation only): returns-shaped DataFrame from
  portfolio.missing_return_sensitivity, run through the same `_headline_stats` path and
  reported alongside (not blended into) the headline.

Zero-variance ratios: Sharpe/IR is NaN ("N/A") whenever the excess/active series has ~zero
std, regardless of its mean (e.g. the benchmark's own Sharpe, and its IR against itself).

All tables/charts are gross of trading costs unless marked net (a `cost_basis` column, a
`_net`/`_gross` suffix, or the chart title).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



REGIMES = {
    '2021': ('2021-01-01', '2021-12-31'),
    '2022': ('2022-01-01', '2022-12-31'),
    '2023-2025': ('2023-01-01', '2025-12-31'),
    '2026': ('2026-01-01', '2026-12-31'),
}

IR_FORMULA = ('IR = sqrt(12) * mean(active) / std(active), '
              'active = total_ret - (TB3MS/1200 + 0.04/12)')

PERIOD_LABEL = f"{config.TEST_START:%m/%Y}–{config.TEST_END:%m/%Y}"
_ANCHOR = config.TEST_START - pd.offsets.MonthEnd(1)  # 2020-12-31: pre-backtest chart anchor
_ZERO_STD_TOL = 1e-8  # below any realistic monthly-return std; floating noise on an exactly
# constant series (e.g. bench_ret - rf_m, or bench_ret vs itself) lands here.


# ---------------------------------------------------------------- low-level stats
def _ratio_or_nan(mean: float, sd: float) -> float:
    """sqrt(12)*mean/sd; undefined (NaN) whenever sd is ~0, never +/-inf."""
    if not np.isfinite(sd) or sd < _ZERO_STD_TOL:
        return np.nan
    return float(np.sqrt(12) * mean / sd)


def ir(active: pd.Series) -> float:
    active = active.dropna()
    return _ratio_or_nan(active.mean(), active.std(ddof=1))


def sharpe(ret: pd.Series, rf: pd.Series) -> float:
    excess = (ret - rf).dropna()
    return _ratio_or_nan(excess.mean(), excess.std(ddof=1))


def cagr(ret: pd.Series) -> float:
    # unreachable in this pipeline: every series reaching here is the (asserted) 68-month test
    # window with missing next-month returns already 0-filled (A4), never empty.
    ret = ret.dropna()
    return float((1 + ret).prod() ** (12 / len(ret)) - 1)


def cumulative(ret: pd.Series) -> float:
    return float((1 + ret.dropna()).prod() - 1)


def drawdown(ret: pd.Series) -> pd.Series:
    """Underwater series; starting capital (wealth=1) counts as a peak, so a month-1 loss is
    already a drawdown instead of being masked by cummax() starting at wealth[0]."""
    wealth = (1 + ret.dropna()).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1


def max_drawdown(ret: pd.Series) -> float:
    # unreachable in this pipeline, same as cagr() above: dd is never empty.
    return float(drawdown(ret).min())


def hit_rate(active: pd.Series) -> float:
    return float((active.dropna() > 0).mean())


def alpha_beta(returns: pd.DataFrame, ret_col: str = 'total_ret') -> dict:
    """OLS of (returns[ret_col] - rf_m) on sp500_exret; plain and Newey-West (3 lag) SEs."""
    y = (returns[ret_col] - returns['rf_m']).astype(float)
    x = sm.add_constant(returns['sp500_exret'].astype(float))
    ols = sm.OLS(y, x, missing='drop').fit()
    nw = sm.OLS(y, x, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3}, use_t=True)
    return {
        'alpha_m': ols.params['const'], 'alpha_ann': ols.params['const'] * 12,
        'beta': ols.params['sp500_exret'],
        'alpha_se_ols': ols.bse['const'], 'alpha_t_ols': ols.tvalues['const'],
        'beta_se_ols': ols.bse['sp500_exret'], 'beta_t_ols': ols.tvalues['sp500_exret'],
        'alpha_se_nw': nw.bse['const'], 'alpha_t_nw': nw.tvalues['const'],
        'beta_se_nw': nw.bse['sp500_exret'], 'beta_t_nw': nw.tvalues['sp500_exret'],
        'n_obs': int(ols.nobs),
    }


def _window_mask(dates) -> pd.Series:
    """Boolean mask for config.TEST_START..config.TEST_END: the one test-window filter, applied
    everywhere the test window is sliced (returns, holdings, gate weight series)."""
    return (dates >= config.TEST_START) & (dates <= config.TEST_END)


def _turnover(returns: pd.DataFrame) -> pd.Series:
    """Turnover excluding `first_month` (empty prior book, not a like-for-like rebalance)."""
    return returns.loc[~returns['first_month'], 'turnover']


def _labels(tbl: pd.DataFrame) -> pd.Series:
    return tbl['ticker'].astype(str) + ', ' + tbl['company_name'].astype(str)


# ---------------------------------------------------------------- tables
def _series_row(ret, active, rf):
    r = ret.dropna()
    return {
        'avg_monthly': ret.mean(), 'ann_arith': ret.mean() * 12, 'cagr': cagr(ret),
        'cumulative': cumulative(ret), 'sharpe': sharpe(ret, rf), 'max_dd': max_drawdown(ret),
        'ir': ir(active) if active is not None else np.nan,
        'hit_rate': hit_rate(active) if active is not None else np.nan,
        'best_month': r.idxmax(), 'best_val': float(r.max()),
        'worst_month': r.idxmin(), 'worst_val': float(r.min()),
    }


def performance_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = {
        'strategy_gross': _series_row(returns['total_ret'], returns['active_ret'], returns['rf_m']),
        'strategy_net': _series_row(returns['total_ret_net'], returns['active_ret_net'], returns['rf_m']),
        'benchmark': _series_row(returns['bench_ret'], None, returns['rf_m']),
        'sp500': _series_row(returns['sp500_ret'], returns['sp500_ret'] - returns['bench_ret'], returns['rf_m']),
    }
    df = pd.DataFrame(rows).T
    for leg, col in [('long_leg_excess_contrib', 'long_ret'), ('short_leg_excess_contrib', 'short_ret')]:
        df.loc[leg, ['avg_monthly', 'ann_arith', 'cumulative']] = [
            returns[col].mean(), returns[col].mean() * 12, cumulative(returns[col])]
    df['cost_basis'] = 'gross'
    df.loc['strategy_net', 'cost_basis'] = 'net'
    for label, col in [('strategy_gross', 'total_ret'), ('strategy_net', 'total_ret_net'),
                        ('benchmark', 'bench_ret'), ('sp500', 'sp500_ret')]:
        df.loc[label, 'corr_sp500'] = returns[col].corr(returns['sp500_ret'])
    return df


def calendar_year_table(returns: pd.DataFrame) -> pd.DataFrame:
    def compound(s):
        return (1 + s).prod() - 1
    g = returns.groupby(returns.index.year)
    out = pd.DataFrame({
        'strategy_gross': g['total_ret'].apply(compound), 'strategy_net': g['total_ret_net'].apply(compound),
        'benchmark': g['bench_ret'].apply(compound), 'sp500': g['sp500_ret'].apply(compound),
    })
    out.index = [f'{y} YTD (Jan–Aug)' if y == 2026 else str(y) for y in out.index]
    return out


def exposure_table(returns: pd.DataFrame, holdings: pd.DataFrame) -> pd.Series:
    turns = _turnover(returns)
    absw = holdings['weight'].abs()

    def top10_share(w):
        w = w.abs()
        tot = w.sum()
        return w.nlargest(10).sum() / tot if tot else np.nan

    top10 = holdings.groupby('month')['weight'].apply(top10_share)

    # A10 diagnostics: how often optimize_month had to relax a tolerance, and the filer-net
    # exposure the SECTOR_TOL-as-a-sector constraint is meant to keep small. The A15 cardinality
    # guard appends a 'cardinality' token to `relax` when it fires (src/portfolio.py); that's a
    # separate mechanism from the RELAX_STEPS tolerance ladder, so it's stripped out here and
    # reported on its own (cardinality_guard_share) rather than counted toward relax_share/
    # relax_count_*, which stay tolerance-relaxation-only.
    relax_raw = returns['relax'].astype(str)
    cardinality_guard = relax_raw.str.contains('cardinality')
    relax_tol = relax_raw.apply(
        lambda s: ','.join(tok for tok in s.split(',') if tok and tok != 'cardinality'))
    relaxed = relax_tol != ''
    relax_counts = relax_tol.loc[relaxed].value_counts()
    relax_stats = {'relax_share': relaxed.mean(), 'cardinality_guard_share': cardinality_guard.mean()}
    relax_stats.update({f'relax_count_{label}': int(n) for label, n in relax_counts.items()})

    return pd.Series({
        'avg_n_long': returns['n_long'].mean(), 'avg_n_short': returns['n_short'].mean(),
        'gross_avg': returns['gross'].mean(), 'gross_min': returns['gross'].min(), 'gross_max': returns['gross'].max(),
        'net_avg': returns['net'].mean(), 'net_min': returns['net'].min(), 'net_max': returns['net'].max(),
        'filer_net_avg': returns['filer_net'].mean(), 'filer_net_min': returns['filer_net'].min(),
        'filer_net_max': returns['filer_net'].max(),
        **relax_stats,
        'avg_abs_weight': absw.mean(), 'max_abs_weight': absw.max(),
        'top10_share_gross_avg': top10.mean(),
        'turnover_avg': turns.mean(), 'turnover_min': turns.min(), 'turnover_max': turns.max(),
        'beta_exante_avg': returns['beta_exante'].mean(),
        'missing_ret_weight_avg': returns['missing_ret_weight'].mean(),
    })


def short_book_table(holdings: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Weighted averages divide by the weight of rows with a non-null value only; the two
    share_* rows divide by total short weight (shares of the whole short book)."""
    shorts = holdings[holdings['weight'] < 0]
    m = shorts.merge(panel[['permno', 'eom', 'me', 'dolvol_126d_raw', 'size_grp']], on=['permno', 'eom'], how='left')
    w = m['weight'].abs()
    m['dolvol_q20'] = m['eom'].map(panel.groupby('eom')['dolvol_126d_raw'].quantile(0.20))

    def wavg(col):
        mask = m[col].notna()
        tw = w[mask].sum()
        return float((w[mask] * m.loc[mask, col]).sum() / tw) if tw else np.nan

    tot_w = w.sum()
    nano_micro = w[m['size_grp'].isin(['nano', 'micro'])].sum() / tot_w if tot_w else np.nan
    bottom_q = w[(m['dolvol_126d_raw'] <= m['dolvol_q20']).fillna(False)].sum() / tot_w if tot_w else np.nan
    return pd.Series({
        'avg_me_weighted': wavg('me'), 'avg_me_simple': m['me'].mean(),
        'avg_dolvol_weighted': wavg('dolvol_126d_raw'), 'avg_dolvol_simple': m['dolvol_126d_raw'].mean(),
        'share_nano_micro_weight': nano_micro,
        'share_bottom_quintile_dolvol_weight': bottom_q,
    })


def contributors(holdings: pd.DataFrame, panel: pd.DataFrame, n: int = 10):
    """Sum over months of weight * stock_exret (gross) by permno; top/bottom n with labels.
    Label is ticker/company_name from the latest `eom` the permno was actually HELD: `holdings`
    only contains held rows, so the groupby-last never picks a later out-of-holding label."""
    m = holdings.merge(panel[['permno', 'eom', 'stock_exret']], on=['permno', 'eom'], how='left')
    m['contrib'] = m['weight'] * m['stock_exret'].fillna(0.0)
    total = m.groupby('permno')['contrib'].sum().rename('total_contrib_gross')
    labels = holdings.sort_values('eom').groupby('permno')[['ticker', 'company_name']].last()
    tbl = labels.join(total).sort_values('total_contrib_gross', ascending=False)
    tbl['label'] = _labels(tbl)
    return tbl.head(n), tbl.tail(n).sort_values('total_contrib_gross')


def top_holdings(holdings: pd.DataFrame, n: int = 10):
    """Top n long / short names by average weight over ALL test months (sum of weight by
    permno / number of test months) -- zero for months a name isn't held. Label is
    ticker/company_name from the latest `eom` the permno was actually HELD (same as
    contributors()), never a later out-of-holding label."""
    n_months = holdings['month'].nunique()
    avg_w = (holdings.groupby('permno')['weight'].sum() / n_months).rename('avg_weight')
    labels = holdings.sort_values('eom').groupby('permno')[['ticker', 'company_name']].last()
    tbl = labels.join(avg_w)
    tbl['label'] = _labels(tbl)
    return tbl.sort_values('avg_weight', ascending=False).head(n), tbl.sort_values('avg_weight').head(n)


def regime_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, (a, b) in REGIMES.items():
        sub = returns.loc[a:b]
        if sub.empty:
            continue
        rows[name] = {
            'ir_gross': ir(sub['active_ret']), 'ann_active_gross': sub['active_ret'].mean() * 12,
            'long_contrib_gross': sub['long_ret'].mean() * 12, 'short_contrib_gross': sub['short_ret'].mean() * 12,
            'n_months': len(sub),
        }
    return pd.DataFrame(rows).T


def neutrality_table(returns: pd.DataFrame) -> pd.DataFrame:
    """The committee's first check: is this book actually market-neutral? One row per metric:
    full-period beta (OLS + Newey-West SE/t-stat, via alpha_beta), per-calendar-year beta (same,
    computed on each calendar year's rows only), average ex-ante beta, average/min/max net
    exposure, and correlation with the S&P. `se`/`t_stat` are populated for the beta rows only."""
    rows = {}
    full = alpha_beta(returns, 'total_ret')
    rows['beta_full_period'] = {'value': full['beta'], 'se': full['beta_se_nw'],
                                 't_stat': full['beta_t_nw'], 'n_obs': full['n_obs']}
    for year, sub in returns.groupby(returns.index.year):
        if len(sub) < 3:
            continue
        yb = alpha_beta(sub, 'total_ret')
        rows[f'beta_{year}'] = {'value': yb['beta'], 'se': yb['beta_se_nw'],
                                 't_stat': yb['beta_t_nw'], 'n_obs': yb['n_obs']}
    extra = {
        'avg_ex_ante_beta': returns['beta_exante'].mean(),
        'avg_net_exposure': returns['net'].mean(),
        'min_net_exposure': returns['net'].min(),
        'max_net_exposure': returns['net'].max(),
        'corr_sp500': returns['total_ret'].corr(returns['sp500_ret']),
    }
    for name, value in extra.items():
        rows[name] = {'value': value, 'se': np.nan, 't_stat': np.nan, 'n_obs': len(returns)}
    return pd.DataFrame(rows).T


def ablation_table(results: dict) -> pd.DataFrame:
    """Each variant is passed through the same test-window restriction/assert as `returns`."""
    rows = {}
    for name, r in results.items():
        r = _restrict_window(r)
        ab = alpha_beta(r)
        rows[name] = {
            'ir': ir(r['active_ret']), 'sharpe': sharpe(r['total_ret'], r['rf_m']),
            'ann_active': r['active_ret'].mean() * 12, 'beta': ab['beta'],
            'turnover': _turnover(r).mean(), 'max_dd': max_drawdown(r['total_ret']),
        }
    return pd.DataFrame(rows).T


def gate_weight_series(gate_coefs: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Effective weight b_k + sum_j c_kj * standardized(s_j(t)) per specialist k, by target
    month: one merge on test_year (vectorised, no per-month/per-specialist loop)."""
    state_vars = config.STATE_VARS
    stat_cols = [f'state_mean_{v}' for v in state_vars] + [f'state_std_{v}' for v in state_vars]
    stats = gate_coefs.groupby('test_year')[stat_cols].first().reset_index()
    wide = gate_coefs.pivot_table(index=['test_year', 'specialist'], columns='term', values='coef').reset_index()

    s = state.reset_index().rename(columns={'index': 'eom'})
    s['target_month'] = s['eom'] + pd.offsets.MonthEnd(1)
    s['test_year'] = s['target_month'].dt.year
    s = s.merge(stats, on='test_year', how='inner')
    for v in state_vars:
        sd = s[f'state_std_{v}'].replace(0, np.nan)
        s[f'z_{v}'] = ((s[v] - s[f'state_mean_{v}']) / sd).fillna(0.0)
    s = s[['target_month', 'test_year'] + [f'z_{v}' for v in state_vars]]

    merged = s.merge(wide, on='test_year', how='inner')
    merged['weight'] = merged['base'].fillna(0.0)
    for v in state_vars:
        merged['weight'] += merged[f'z_{v}'] * merged[v].fillna(0.0)

    out = merged.pivot_table(index='target_month', columns='specialist', values='weight')
    return out.loc[_window_mask(out.index)].sort_index()


# ---------------------------------------------------------------- charts
def _savefig(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=150)
    plt.close(fig)


def _title(base: str) -> str:
    return f"{base}, gross of trading costs, {PERIOD_LABEL}"


def _anchored(s: pd.Series) -> pd.Series:
    """Prepend a zero value at the pre-backtest anchor month (2020-12-31)."""
    return pd.concat([pd.Series([0.0], index=[_ANCHOR]), s])


def _line_plot(series_dict: dict, title: str, fname: str, hline=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, s in series_dict.items():
        ax.plot(s.index, s.values, label=label)
    if hline is not None:
        ax.axhline(hline, color='black', lw=0.8)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series_dict) > 1:
        ax.legend()
    _savefig(fig, fname)


def plot_cumulative(returns):
    series = {label: _anchored((1 + returns[col]).cumprod() - 1)
              for col, label in [('total_ret', 'Strategy'), ('bench_ret', 'Benchmark'), ('sp500_ret', 'S&P 500')]}
    _line_plot(series, _title('Cumulative return'), 'cumulative_returns.png', ylabel='Cumulative return')


def plot_underwater(returns):
    series = {'Strategy': _anchored(drawdown(returns['total_ret'])),
              'S&P 500': _anchored(drawdown(returns['sp500_ret']))}
    _line_plot(series, _title('Drawdown'), 'underwater.png', ylabel='Drawdown')


def plot_rolling_active(returns, window: int = 12):
    roll = returns['active_ret'].rolling(window).mean() * 12
    _line_plot({'Active return (ann.)': roll},
               _title(f'Rolling {window}-month annualized active return'), 'rolling_active_return.png', hline=0)


def plot_rolling_ir(returns, window: int = 12):
    m = returns['active_ret'].rolling(window).mean()
    sd = returns['active_ret'].rolling(window).std(ddof=1)
    _line_plot({'IR': np.sqrt(12) * m / sd},
               _title(f'Rolling {window}-month information ratio'), 'rolling_ir.png', hline=0)


def plot_rolling_beta(returns, window: int = 12):
    y = returns['total_ret'] - returns['rf_m']
    x = returns['sp500_exret']
    roll_beta = y.rolling(window).cov(x) / x.rolling(window).var()
    _line_plot({'Beta': roll_beta}, _title(f'Rolling {window}-month beta vs S&P 500'), 'rolling_beta.png', hline=0)


def plot_return_histogram(returns):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(returns['total_ret'], bins=20, color='tab:blue', alpha=0.85)
    ax.axvline(returns['bench_ret'].mean(), color='red', linestyle='--', label='Avg monthly hurdle')
    ax.set_title(_title('Distribution of monthly returns'))
    ax.legend()
    _savefig(fig, 'return_histogram.png')


def plot_contributors(top, bottom):
    both = pd.concat([top, bottom])
    colors = ['tab:green' if v >= 0 else 'tab:red' for v in both['total_contrib_gross']]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(both['label'], both['total_contrib_gross'], color=colors)
    ax.invert_yaxis()
    ax.set_title(_title('Top/bottom return contributors (excess-return contribution)'))
    _savefig(fig, 'contributors.png')


def plot_beta_by_year(neutrality: pd.DataFrame):
    """Bar chart of per-calendar-year beta vs S&P 500 (NW SE error bars), from neutrality_table's
    beta_YYYY rows -- the committee's first check, at a glance."""
    yearly = neutrality[neutrality.index.str.startswith('beta_') & (neutrality.index != 'beta_full_period')]
    years = [idx.replace('beta_', '') for idx in yearly.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(years, yearly['value'], yerr=yearly['se'], color='tab:blue', capsize=4)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Beta vs S&P 500')
    ax.set_title(f'Beta by calendar year, {PERIOD_LABEL}')
    _savefig(fig, 'beta_by_year.png')


def plot_gate_weights(gate_coefs, state):
    series = gate_weight_series(gate_coefs, state)
    _line_plot({c: series[c] for c in series.columns},
               f'Gate effective specialist weights, {PERIOD_LABEL}', 'gate_weights.png')


# ---------------------------------------------------------------- orchestration
def _restrict_window(returns: pd.DataFrame) -> pd.DataFrame:
    out = returns.loc[_window_mask(returns.index)].sort_index()
    missing = pd.date_range(config.TEST_START, config.TEST_END, freq='ME').difference(out.index)
    assert len(missing) == 0, f'evaluate: missing test-window months: {[d.date() for d in missing]}'
    return out


def _headline_stats(returns: pd.DataFrame, perf: pd.DataFrame, ab: dict) -> dict:
    """Built entirely from the performance table and alpha/beta results -- no recomputation."""
    return {
        'ir': perf.loc['strategy_gross', 'ir'], 'sharpe': perf.loc['strategy_gross', 'sharpe'],
        'cagr': perf.loc['strategy_gross', 'cagr'], 'max_dd': perf.loc['strategy_gross', 'max_dd'],
        'ir_net': perf.loc['strategy_net', 'ir'],
        'alpha_ann': ab['alpha_ann'], 'alpha_t_nw': ab['alpha_t_nw'],
        'beta': ab['beta'], 'beta_se_nw': ab['beta_se_nw'], 'beta_t_nw': ab['beta_t_nw'],
        'avg_turnover': _turnover(returns).mean(),
        'avg_n_long': returns['n_long'].mean(), 'avg_n_short': returns['n_short'].mean(),
    }


def run_evaluation(returns, holdings, panel, gate_coefs=None, state=None, ablations=None,
                    sensitivity_returns=None) -> dict:
    """sensitivity_returns: optional, same schema as `returns`, from
    portfolio.missing_return_sensitivity -- run through the same _headline_stats path as the
    headline, reported alongside it (not blended into it)."""
    returns = _restrict_window(returns)
    holdings = holdings[_window_mask(holdings['month'])]
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    print(IR_FORMULA)

    # neutrality check first -- this is the committee's first question, so it's the first table
    # computed, written and printed, ahead of performance/exposure/everything else.
    neutrality = neutrality_table(returns)
    neutrality.to_csv(config.TABLE_DIR / 'neutrality_table.csv')
    nf = neutrality.loc['beta_full_period']
    print('=== NEUTRALITY CHECK (committee first look) ===')
    print(f"full-period beta={nf['value']:.3f} (NW t-stat={nf['t_stat']:.2f}), "
          f"avg ex-ante beta={neutrality.loc['avg_ex_ante_beta', 'value']:.3f}, "
          f"avg net exposure={neutrality.loc['avg_net_exposure', 'value']:.3f} "
          f"(min={neutrality.loc['min_net_exposure', 'value']:.3f}, "
          f"max={neutrality.loc['max_net_exposure', 'value']:.3f}), "
          f"corr(S&P 500)={neutrality.loc['corr_sp500', 'value']:.3f}")

    perf = performance_table(returns)
    perf.to_csv(config.TABLE_DIR / 'performance_table.csv')
    ab_gross = alpha_beta(returns, 'total_ret')
    ab_net = alpha_beta(returns, 'total_ret_net')
    pd.DataFrame([ab_gross, ab_net], index=['gross', 'net']).to_csv(config.TABLE_DIR / 'alpha_beta.csv')
    calendar_year_table(returns).to_csv(config.TABLE_DIR / 'calendar_year_table.csv')
    exposure_table(returns, holdings).to_csv(config.TABLE_DIR / 'exposure_table.csv')
    short_book_table(holdings, panel).to_csv(config.TABLE_DIR / 'short_book_table.csv')
    top_c, bot_c = contributors(holdings, panel)
    pd.concat([top_c, bot_c]).to_csv(config.TABLE_DIR / 'contributors.csv')
    long_h, short_h = top_holdings(holdings)
    pd.concat([long_h, short_h], keys=['long', 'short']).to_csv(config.TABLE_DIR / 'top_holdings.csv')
    regime_table(returns).to_csv(config.TABLE_DIR / 'regime_table.csv')
    if ablations:
        ablation_table(ablations).to_csv(config.TABLE_DIR / 'ablation_table.csv')

    base_stats = _headline_stats(returns, perf, ab_gross)
    headline = {**base_stats, 'ir_formula': IR_FORMULA}

    if sensitivity_returns is not None:
        sensitivity_returns = _restrict_window(sensitivity_returns)
        sens_perf = performance_table(sensitivity_returns)
        sens_ab = alpha_beta(sensitivity_returns, 'total_ret')
        sens_stats = _headline_stats(sensitivity_returns, sens_perf, sens_ab)
        pd.DataFrame([base_stats, sens_stats], index=['headline', 'sensitivity']).to_csv(
            config.TABLE_DIR / 'sensitivity_table.csv')
        headline.update({f'sensitivity_{k}': v for k, v in sens_stats.items()})

    plot_cumulative(returns)
    plot_underwater(returns)
    plot_rolling_active(returns)
    plot_rolling_ir(returns)
    plot_rolling_beta(returns)
    plot_return_histogram(returns)
    plot_contributors(top_c, bot_c)
    plot_beta_by_year(neutrality)
    if gate_coefs is not None and state is not None:
        plot_gate_weights(gate_coefs, state)

    return headline
''', config=config)

# ===== MAIN.py =====
"""AlphaBERT -- end-to-end pipeline entry point (owns this file only; see docs/SPEC.md section 9/10).

    python MAIN.py                  # full run: recompute predictions, then backtest/evaluate/submit
    python MAIN.py --reuse-preds    # reuse outputs/cache/{preds,gate_coefs}.parquet from a prior run
    python MAIN.py --dry            # load cached panel/text/market, print shapes, exit (no fit/backtest)

Cache-aware throughout (src/data.py, src/text.py): an empty outputs/cache/ reproduces the whole
pipeline from the raw parquet files in data/.

Steps (each a thin call into the owning src module -- no modeling/portfolio logic lives here):
  1. Build the panel, market-state series and external market data (T-bill, S&P 500).
  2. Attach text (8-K) features, then geometry features (src/geometry.py: FinBERT embeddings ->
     PCA -> k-means event types; a no-op if embeddings are missing). Tone columns are included
     only when the settings-specific FinBERT score cache (text.scores_path_for(text.MAX_LENGTH))
     is a complete run.
  3. models.run_all(): baselines, the five A9 characteristic specialists + text, the ridge gate,
     the equal-weight/lgbm_all/blend combiners, and pred_auto (per test year, the walk-forward
     choice among config.HEADLINE_CANDIDATES with the best mean IC on that year's own validation
     window) -- per test year -> OOS R2/IC table.
  4. Calibrate the optimizer's two penalties ONCE on the SMOOTHED pred_ew signal from the valid
     rows of the FIRST test year only (config.TEST_YEARS[0], the 2019-2020 window -- preds now
     carries valid rows for every test year, so this filter is required), then lock them.
  5. Headline backtest: config.HEADLINE_SIGNAL (pred_auto), the neutral optimizer (A14
     sector-demeaned candidates, NET_MODE='beta' by default -- see src/portfolio.py
     optimize_month) over formation months 2020-12..2026-07, labels, and an adverse
     missing-return sensitivity check.
  6. Same locked penalties/smoothing for every ablation signal: pred_ew, pred_lgbm_all,
     pred_blend, pred_gate, pred_gate_notext, pred_ew_notext, pred_ridge, and every
     pred_spec_* -- whichever of these isn't the headline column itself (the headline is
     never backtested twice).
  7. evaluate.run_evaluation() for tables/charts (including the neutrality_table -- the
     committee's first check); write the 3-file submission.
  8. Print the headline performance dict.
"""
import argparse
from pathlib import Path

import pandas as pd




def run_signal(col, test_preds, panel, market, l2, tc):
    """Smooth preds column `col` into a signal and run the neutral optimizer backtest over it.
    Shared by the headline (config.HEADLINE_SIGNAL) and every ablation signal so they use
    identical smoothing and locked penalties."""
    sig = test_preds[['permno', 'eom']].copy()
    sig['signal'] = portfolio.smooth(test_preds, col)
    return portfolio.backtest(sig, panel, market, l2=l2, tc=tc)


def ablation_columns(headline_signal, specialist_cols):
    """Every ablation signal for step 6: pred_ew, pred_lgbm_all, pred_blend, pred_gate,
    pred_gate_notext, pred_ew_notext, pred_ridge -- whichever of these isn't `headline_signal`
    -- plus every pred_spec_* specialist column, sorted. Never includes `headline_signal` itself
    (the headline column is never backtested twice)."""
    fixed_cols = ['pred_ew', 'pred_lgbm_all', 'pred_blend', 'pred_gate', 'pred_gate_notext',
                  'pred_ew_notext', 'pred_ridge']
    fixed_ablations = [c for c in fixed_cols if c != headline_signal]
    return fixed_ablations + sorted(specialist_cols)


def headline_r2_row(r2, headline_signal):
    """(model_col, oos_r2) for the headline's row in `r2` (models.r2_table output), or None if
    that column isn't present. pred_ew stays z-score-valued: its return-unit twin pred_ew_ret
    carries the R2 instead. Any other headline (e.g. pred_auto, pred_gate) is itself return-unit
    and has its own row -- no substitution needed."""
    col = 'pred_ew_ret' if headline_signal == 'pred_ew' else headline_signal
    if col not in r2['model'].values:
        return None
    return col, r2.loc[r2['model'] == col, 'oos_r2'].iloc[0]


def main():
    parser = argparse.ArgumentParser(description='AlphaBERT end-to-end pipeline')
    parser.add_argument('--reuse-preds', action='store_true',
                         help='reuse cached outputs/cache/preds.parquet + gate_coefs.parquet instead of '
                              'recomputing models.run_all() (default: always recompute)')
    parser.add_argument('--dry', action='store_true',
                         help='load the cached panel, text features and market data, print shapes, then exit '
                              '(no model fitting, backtest, evaluation or submission)')
    args = parser.parse_args()

    # ---- 1. panel, market state, external market data -----------------------------------------
    print('=== step 1/8: build panel, market state, market data ===')
    panel = data.build_panel()
    state = data.market_state()
    market = data.load_market()

    # ---- 2. text + geometry features --------------------------------------------------------
    print('=== step 2/8: text + geometry features ===')
    panel = text.add_text_features(panel)
    scores_path = text.scores_path_for(text.MAX_LENGTH)
    text_cols_used = [c for c in text.TEXT_FEATURES if c in panel.columns]
    tone_included = {'tone_mean', 'tone_min', 'fb_neg_max'}.issubset(set(text_cols_used))
    print(f'FinBERT cache ({scores_path.name}) present: {scores_path.exists()}; tone included: {tone_included}')
    print(f'text features in use: {text_cols_used}')


    cols_before_geom = set(panel.columns)
    panel = geometry.add_geometry_features(panel)
    geom_cols = sorted(set(panel.columns) - cols_before_geom)
    print(f'geometry features present: {len(geom_cols)} ({geom_cols})')

    if args.dry:
        print('=== --dry: skipping model fit / backtest / evaluate / submission ===')
        print(f'panel shape: {panel.shape}')
        print(f'market_state shape: {state.shape}')
        print(f'market shape: {market.shape}')
        return None

    # ---- 3. models: baselines, specialists, combiners, pred_auto, per test year ----------------
    print('=== step 3/8: models.run_all (or reuse cached preds) ===')
    preds_path = config.CACHE_DIR / 'preds.parquet'
    if args.reuse_preds and preds_path.exists():
        preds = pd.read_parquet(preds_path)
        for c in ('eom', 'target_month'):
            if c in preds.columns:
                preds[c] = pd.to_datetime(preds[c]).astype('datetime64[ns]')
        missing = set(models.PRED_COLS) - set(preds.columns)
        models_path = Path(models.__file__)
        if missing:
            raise RuntimeError(
                f'--reuse-preds: cached {preds_path} is missing columns {sorted(missing)} that '
                f'models.PRED_COLS now expects -- this cache was built by an older src/models.py. '
                f'Re-run without --reuse-preds to rebuild it.')
        if preds_path.stat().st_mtime < models_path.stat().st_mtime:
            raise RuntimeError(
                f'--reuse-preds: cached {preds_path} is older than src/models.py -- the cache '
                f'predates the current model code. Re-run without --reuse-preds to rebuild it.')
        print(f'--reuse-preds set: loading cached {preds_path}')
    else:
        preds = models.run_all(panel, state)

    r2 = models.r2_table(preds)
    print('OOS R2 / IC table (test rows):')
    print(r2.to_string(index=False))
    headline_r2 = headline_r2_row(r2, config.HEADLINE_SIGNAL)
    if headline_r2 is not None:
        col, val = headline_r2
        print(f'headline OOS R2 ({col}, config.HEADLINE_SIGNAL={config.HEADLINE_SIGNAL}): {val}')
    r2.to_csv(config.TABLE_DIR / 'oos_r2.csv', index=False)

    # ---- 4. calibrate portfolio penalties on the SMOOTHED first-test-year validation pred_ew ---
    print('=== step 4/8: calibrate portfolio penalties (2019-2020 validation, smoothed pred_ew) ===')
    valid_mask = (preds['split'] == 'valid') & (preds['test_year'] == config.TEST_YEARS[0])
    valid_ew = preds.loc[valid_mask, ['permno', 'eom', 'pred_ew']].copy()
    valid_ew['signal'] = portfolio.smooth(valid_ew, 'pred_ew')
    valid_ew = valid_ew[['permno', 'eom', 'signal']]
    l2, tc, calib_table = portfolio.calibrate(valid_ew, panel, market)
    print(f'locked penalties: L2_PENALTY={l2}, TURNOVER_PENALTY={tc}')
    print(calib_table.to_string(index=False))
    calib_table.to_csv(config.TABLE_DIR / 'calibration.csv', index=False)

    # ---- 5. headline backtest: config.HEADLINE_SIGNAL, smoothed, formation months 2020-12..2026-07
    print(f'=== step 5/8: headline backtest ({config.HEADLINE_SIGNAL}) ===')
    test_preds = preds.loc[preds['split'] == 'test']

    holdings, returns = run_signal(config.HEADLINE_SIGNAL, test_preds, panel, market, l2, tc)
    label_panel, filing_labels = portfolio.load_label_sources()
    holdings = portfolio.attach_labels(holdings, label_panel, filing_labels)
    sensitivity = portfolio.missing_return_sensitivity(holdings, panel, returns)

    # ---- 6. ablations: same locked penalties, same smoothing -------------------------------------
    print('=== step 6/8: ablations ===')
    specialist_cols = [c for c in test_preds.columns if c.startswith('pred_spec_')]
    ablation_cols = ablation_columns(config.HEADLINE_SIGNAL, specialist_cols)
    ablations = {}
    ablation_failures = []
    for col in ablation_cols:
        try:
            _, ab_returns = run_signal(col, test_preds, panel, market, l2, tc)
        except RuntimeError as e:
            print(f'  ablation FAILED (infeasible under neutrality constraints): {col}: {e}')
            ablation_failures.append(dict(signal=col, error=str(e)))
            continue
        ablations[col] = ab_returns
        print(f'  ablation done: {col}')
    pd.DataFrame(ablation_failures, columns=['signal', 'error']).to_csv(
        config.TABLE_DIR / 'ablation_failures.csv', index=False)

    # ---- 7. evaluate + write submission ---------------------------------------------------------
    print('=== step 7/8: evaluate ===')
    gate_coefs = pd.read_parquet(config.CACHE_DIR / 'gate_coefs.parquet')
    headline = evaluate.run_evaluation(returns, holdings, panel, gate_coefs=gate_coefs, state=state,
                                        ablations=ablations, sensitivity_returns=sensitivity)

    print('=== step 8/8: write submission ===')
    portfolio.write_submission(holdings, returns)

    print('=== headline ===')
    print(headline)
    return headline


if __name__ == '__main__':
    main()
