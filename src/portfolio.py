"""Signal smoothing, optimizer, backtest accounting, submission files. See docs/SPEC.md section 6."""
import numpy as np
import pandas as pd
import cvxpy as cp

from src import config

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


def _dust_and_rescale(wv, is_long, is_short):
    """Zero-out dust (|w| < DUST), then rescale each leg back to exactly +1/-1."""
    wv = wv.copy()
    wv[np.abs(wv) < DUST] = 0.0
    for mask, target in ((is_long, 1.0), (is_short, -1.0)):
        idx = np.where(mask)[0]
        if idx.size and wv[idx].sum() != 0:
            wv[idx] = wv[idx] * (target / wv[idx].sum())
    return wv


def _tol_groups(sector, filer):
    """Exposure groups sharing SECTOR_TOL and the sector relaxation step: each GICS2
    sector, plus the has_filing==1 group -- the filer-net-neutral constraint (A10: 8-K
    coverage encodes future survival, so filers must not be pushed into the signal tails
    as a group; has_filing is mandatory, see A10/`optimize_month`)."""
    groups = {f'sector {s}': sector == s for s in np.unique(sector)}
    groups['filer'] = filer == 1
    return groups


def _check_constraints(w, is_long, is_short, beta, size_z, groups, cap,
                        sector_tol, size_tol, beta_tol, tol=1e-5):
    """One assert block for every optimize_month constraint, at `tol`. `sector_tol`/
    `size_tol`/`beta_tol` are the EFFECTIVE tolerances actually used for the solve that
    produced `w` (a RELAX_STEPS ladder step may have widened them beyond the config base
    values) -- checking against the fixed config constants regardless of which step solved
    would fire spuriously whenever solve_ladder needed to relax. Legs summing to +-1, the
    MAX_WEIGHT cap, and the 100..500 name count are competition rules at fixed values and are
    never relaxed, so they stay checked against fixed constants."""
    assert abs(w[is_long].sum() - 1.0) <= tol, "long leg does not sum to 1"
    assert abs(w[is_short].sum() + 1.0) <= tol, "short leg does not sum to -1"
    assert np.abs(w).max() <= cap + tol, "MAX_WEIGHT breached"
    assert abs(beta @ w) <= beta_tol + tol, "beta exposure breached"
    assert abs(size_z @ w) <= size_tol + tol, "size exposure breached"
    for name, mask in groups.items():
        assert abs(w[mask].sum()) <= sector_tol + tol, f"{name} exposure breached"
    n_names = int((w != 0).sum())
    assert 100 <= n_names <= 500, f"n_names={n_names} out of [100,500]"


def optimize_month(m, w_prev, l2=None, tc=None):
    """m: DataFrame(permno, signal, beta, gics2, size_z, has_filing) for one formation
    month. has_filing is mandatory (A10): the filer group (has_filing==1) gets the same
    |exposure| <= SECTOR_TOL constraint as a sector, filer-net-neutral.
    w_prev: Series(permno -> weight) from the prior month (empty for the first month).

    A14: candidate selection and the objective use the signal DEMEANED WITHIN GICS2 for
    this month (pure within-sector stock selection) -- the raw top/bottom N_CAND by
    cross-sectional signal can be sector-lopsided enough that no weighting satisfies the
    sector-neutrality ladder even at its most relaxed step; under sector-neutral
    constraints the objective is (nearly) invariant to a per-sector shift of the signal,
    so this makes candidate sets sector-balanced at ~no cost to the objective.

    Returns (weights: Series(permno -> weight), relax: str naming which tolerances
    were relaxed to find a feasible solution, '' if none were needed)."""
    l2 = config.L2_PENALTY if l2 is None else l2
    tc = config.TURNOVER_PENALTY if tc is None else tc

    m = m.dropna(subset=['signal']).drop_duplicates('permno').set_index('permno')
    s = m['signal'] - m.groupby('gics2')['signal'].transform('mean')
    long_cand = s.nlargest(config.N_CAND).index
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

    is_long = names.isin(long_cand)
    is_short = names.isin(short_cand)
    is_other = ~(is_long | is_short)
    cap = config.MAX_WEIGHT
    n = len(names)

    def build_constraints(w, sector_tol, size_tol, beta_tol, fixed_zero):
        # is_long/is_short are never all-False: they come from nlargest/nsmallest(N_CAND)
        # over a dropna'd signal, so as long as m has >=1 row each leg is non-empty (and the
        # n_names assert below requires >=100 anyway). is_other can legitimately be empty
        # (e.g. first month, no candidates left over from w_prev), so that guard stays.
        cons = [w[is_long] >= 0, w[is_long] <= cap, cp.sum(w[is_long]) == 1,
                w[is_short] <= 0, w[is_short] >= -cap, cp.sum(w[is_short]) == -1]
        if is_other.any():
            cons.append(w[is_other] == 0)
        cons.append(cp.abs(beta @ w) <= beta_tol)
        cons.append(cp.abs(size_z @ w) <= size_tol)
        for mask in groups.values():
            cons.append(cp.abs(cp.sum(w[mask])) <= sector_tol)
        if fixed_zero is not None and fixed_zero.any():
            cons.append(w[fixed_zero] == 0)
        return cons

    def solve_ladder(fixed_zero=None):
        w = cp.Variable(n)
        objective = cp.Maximize(signal @ w - tc * cp.norm1(w - wprev_vec) - l2 * cp.sum_squares(w))
        for sf, zf, bf, label in RELAX_STEPS:
            sector_tol, size_tol, beta_tol = config.SECTOR_TOL * sf, config.SIZE_TOL * zf, config.BETA_TOL * bf
            cons = build_constraints(w, sector_tol, size_tol, beta_tol, fixed_zero)
            if _solve(cp.Problem(objective, cons)):
                if label:
                    print(f"optimize_month: relaxed tolerances {label}")
                return np.asarray(w.value).ravel(), label, (sector_tol, size_tol, beta_tol)
        raise RuntimeError("optimize_month: solver failed even after relaxation")

    wv, relax, tols = solve_ladder()
    dust_mask = np.abs(wv) < DUST
    wv = _dust_and_rescale(wv, is_long, is_short)
    try:
        _check_constraints(wv, is_long, is_short, beta, size_z, groups, cap, *tols)
    except AssertionError:
        # rescaling the dusted solution broke a constraint: re-solve once with
        # the dusted names fixed at exactly 0, then dust/rescale again. Check against
        # THIS solve's effective tolerances (tols), not the first solve's -- the ladder
        # step that actually produced the returned weights may differ between the two calls.
        wv, relax2, tols = solve_ladder(fixed_zero=dust_mask)
        relax = ','.join(x for x in (relax, relax2) if x)
        wv = _dust_and_rescale(wv, is_long, is_short)
        _check_constraints(wv, is_long, is_short, beta, size_z, groups, cap, *tols)

    weights = pd.Series(wv, index=names)
    weights = weights[weights != 0]
    return weights, relax


# ---------------------------------------------------------------- accounting
def compute_month_return(weights, w_prev, ret, rf_m, sp500_ret, sp500_exret, beta_map):
    """weights, w_prev, ret, beta_map: Series indexed by permno. ret may contain NaN
    for missing realized returns (treated as 0, tracked in missing_ret_weight).
    turnover/cost are target-to-target: 0.5*sum|w_t - w_prev| vs the PREVIOUS month's
    target weights, not intra-month drifted weights."""
    idx = weights.index
    r = ret.reindex(idx)
    missing = r.isna()
    r = r.fillna(0.0)

    long_mask = weights > 0
    short_mask = weights < 0
    long_ret = float((weights[long_mask] * r[long_mask]).sum())
    short_ret = float((weights[short_mask] * r[short_mask]).sum())
    ls_ret = long_ret + short_ret
    total_ret = rf_m + ls_ret
    bench_ret = rf_m + config.HURDLE_ANNUAL / 12
    active_ret = total_ret - bench_ret

    all_names = idx.union(w_prev.index)
    dw = weights.reindex(all_names).fillna(0.0) - w_prev.reindex(all_names).fillna(0.0)
    turnover = 0.5 * dw.abs().sum() / config.GROSS
    cost = config.COST_BPS / 1e4 * dw.abs().sum()

    beta_exante = float((weights * beta_map.reindex(idx).fillna(0.0)).sum())

    return dict(
        long_ret=long_ret, short_ret=short_ret, ls_ret=ls_ret, rf_m=rf_m,
        total_ret=total_ret, bench_ret=bench_ret, active_ret=active_ret,
        sp500_ret=sp500_ret, sp500_exret=sp500_exret,
        n_long=int(long_mask.sum()), n_short=int(short_mask.sum()),
        gross=float(weights.abs().sum()), net=float(weights.sum()),
        beta_exante=beta_exante, turnover=float(turnover), cost=float(cost),
        total_ret_net=total_ret - cost, active_ret_net=active_ret - cost,
        missing_ret_weight=float(weights[missing].abs().sum()),
    )


# ---------------------------------------------------------------- backtest
def backtest(signal_df, panel, market, l2=None, tc=None):
    months = sorted(signal_df['eom'].unique())
    assert 'has_filing' in panel.columns, \
        "backtest: panel lacks has_filing (mandatory, A10 filer-net-neutral constraint)"
    aux_cols = ['permno', 'eom', 'beta', 'gics2', 'size_z', 'ticker', 'company_name', 'stock_exret', 'has_filing']
    aux = panel[aux_cols].drop_duplicates(['permno', 'eom'])
    m_cols = ['permno', 'signal', 'beta', 'gics2', 'size_z', 'has_filing']

    holdings_frames = []
    returns_rows = []
    w_prev = pd.Series(dtype=float)
    for i, mth in enumerate(months):
        sig = signal_df.loc[signal_df['eom'] == mth, ['permno', 'eom', 'signal']]
        m = sig.merge(aux, on=['permno', 'eom'], how='inner')
        assert m[['beta', 'gics2', 'size_z']].notna().all().all(), \
            f"backtest: NaN in beta/gics2/size_z after inner-merge with panel aux for {mth}"

        w, relax = optimize_month(m[m_cols], w_prev, l2=l2, tc=tc)

        mi = m.set_index('permno')
        ret = mi['stock_exret']
        beta_map = mi['beta']

        holding_month = mth + pd.offsets.MonthEnd(1)
        if holding_month not in market.index:
            raise ValueError(f"backtest: holding month {holding_month} missing from market (no silent NaN rf_m)")
        row = market.loc[holding_month]
        rf_m, sp500_ret, sp500_exret = row['rf_m'], row['sp500_ret'], row['sp500_exret']

        rec = compute_month_return(w, w_prev, ret, rf_m, sp500_ret, sp500_exret, beta_map)
        rec['filer_net'] = float((w * mi['has_filing'].reindex(w.index).fillna(0)).sum())
        rec['month'] = holding_month
        rec['first_month'] = (i == 0)
        rec['relax'] = relax
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
    out['total_ret'] = out['rf_m'] + out['ls_ret']
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
    label_panel['eom'] = pd.to_datetime(label_panel['eom'])
    filing_labels = pd.read_parquet(config.FILINGS_PATH, columns=['permno', 'filing_date', 'ticker', 'company_name'])
    filing_labels['filing_date'] = pd.to_datetime(filing_labels['filing_date'])
    return label_panel, filing_labels


def _asof_fill(missing, source, date_col):
    """For each (permno, eom) in `missing`, find the most recent labelled row of
    `source` with date_col <= eom, per permno. Returns a DataFrame indexed by
    `missing`'s `_idx` with columns ticker, company_name (rows with no match dropped)."""
    src = source.dropna(subset=['ticker', 'company_name']).sort_values(date_col)
    src = src.rename(columns={date_col: 'eom'})[['permno', 'eom', 'ticker', 'company_name']]
    left = missing.sort_values('eom')
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
    """Round WEIGHT (percent) to 6dp so each month's long leg sums to exactly
    +100.000000 and short leg to -100.000000, capped at `cap`; the rounding
    residual goes on the largest-|weight| name of each leg, only if that keeps
    it within `cap`."""
    out = weight_pct.copy()
    for _, idx in weight_pct.groupby(date).groups.items():
        w = weight_pct.loc[idx]
        for leg_mask, target in ((w > 0, 100.0), (w < 0, -100.0)):
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
    *_net columns are the cost-adjusted companions, included for reference."""
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
        assert abs(g[g > 0].sum() - 100.0) < 1e-6, f"{d}: long WEIGHT does not sum to 100"
        assert abs(g[g < 0].sum() + 100.0) < 1e-6, f"{d}: short WEIGHT does not sum to -100"
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
