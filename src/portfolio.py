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
