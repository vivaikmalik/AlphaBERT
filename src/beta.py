"""Point-in-time stock-beta estimates + uncertainty (docs/research_log.md, A15/A16 postmortem):
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

from src import config

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

    from src import data  # local import: avoids a data.py <-> beta.py circular import at load time
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
    from src import data
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
    from src import data
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
