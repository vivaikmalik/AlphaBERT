# Chapter 4 — Portfolio Construction

This chapter covers `src/portfolio.py` end to end: turning a raw per-stock forecast into a
smoothed signal, the sector-demeaning trick that makes the optimizer solvable, the convex
program that turns a signal into a market-neutral book, the two-parameter calibration that
tunes it, the backtest accounting that turns weights into competition-comparable returns, the
label pipeline, and the three submission CSVs. Every claim below is checked against the code
(`src/portfolio.py`, `src/config.py`), the shared contract (`docs/SPEC.md` §6, §10 amendments
A3, A4, A5, A10, A14), the design decisions and bug diagnosis in `docs/research_log.md`, and the
tests in `tests/test_portfolio.py` / `tests/test_integrity.py`. No backtest has been run yet —
every number in the worked example of §4.8 is a hand-built toy, not a reported result.

Where `portfolio.py` is called from: `MAIN.py` step 4 calibrates the two optimizer penalties
once, on the smoothed 2019–2020 validation `pred_ew` signal (`MAIN.py:main`, "step 4/8"), then
step 5 runs the headline backtest and steps 6–8 rerun the same locked penalties for every
ablation signal, attach labels, compute the adverse sensitivity, and write the submission
(`MAIN.py:run_signal`, `MAIN.py:main`).

## 4.1 From forecast to signal: `smooth()`

A model produces one forecast column per stock-month (e.g. `pred_ew`). Before that forecast can
drive an optimizer, `smooth()` (`src/portfolio.py:smooth`) turns it into a signal in two steps.

**Step 1 — cross-sectional z-score within the formation month.** For every `eom` group:

$$
z_{i,t} = \frac{x_{i,t} - \mu_t}{\sigma_t}, \qquad \mu_t = \text{mean}_i(x_{i,t}),\ \ \sigma_t = \text{std}_i(x_{i,t})
$$

with $z_{i,t} = 0$ if $\sigma_t = 0$ or undefined (code: `mu = grp.transform('mean')`,
`sd = grp.transform('std', ddof=0)`, `.where(sd > 0, 0.0).fillna(0.0)`). This makes the signal
comparable in scale across months regardless of the raw forecast's units.

**Step 2 — per-permno EMA across formation months, past-only, reset after a gap.** Sorting by
`(permno, eom)` and walking forward one row at a time:

$$
s_{i,t} = \begin{cases}
z_{i,t} & \text{if } t \text{ is permno } i\text{'s first observed month, or the previous
observed month for } i \text{ was not } t-1 \\[4pt]
\alpha\, z_{i,t} + (1-\alpha)\, s_{i,t-1} & \text{otherwise}
\end{cases}
$$

with `config.EMA_ALPHA` $= \alpha = 0.5$. The gap check compares consecutive integer month
keys (`mkey = eom.year*12 + eom.month`); if `permno[i] != prev_permno` or
`mkey[i] != prev_mkey + 1`, the EMA resets to the raw z-score rather than blending with a stale
`s_prev` from a month that isn't actually adjacent. This is exactly what
`test_smooth_past_only_and_reset` checks: permno 1 has eom 2020-01, 02, 03, then skips April and
reappears in May; May's smoothed value equals May's own z-score, not $\alpha z_{May} + (1-\alpha)
s_{Mar}$.

**Why an EMA at all — turnover.** The optimizer's objective (§4.4) penalizes $\lVert w -
w_{prev}\rVert_1$, i.e. trading. A raw forecast can flip sign or magnitude from month to month
purely from estimation noise; smoothing the input to the optimizer, rather than relying on the
turnover penalty alone to suppress churn, reduces how often a name's *rank* crosses a threshold
that would force a trade. `docs/research_log.md`'s pre-registration entry records the design
choice explicitly: "no hold buffer (EMA + turnover penalty suffice)".

**Past-only property.** `smooth()` only ever reads $z_{i,\tau}$ for $\tau \le t$ when computing
$s_{i,t}$ — the recursion runs strictly forward in time and never looks at a later `eom`. The
test confirms this operationally: mutating the March value and recomputing leaves January and
February's smoothed output bit-for-bit identical (`df2`/`s2` in
`test_smooth_past_only_and_reset`). This matters because `smooth()` is later applied to the full
`test_preds` frame in `MAIN.py:run_signal`, so a look-ahead bug here would leak future
information into every formation month's signal. `MAIN.py` smooths the *concatenated* test
predictions across test-year boundaries, so the EMA blends forecasts from consecutive annual model
vintages; that is safe because each vintage is fitted only on data before its own test year, so
blending across the boundary carries no look-ahead.

## 4.2 Sector demeaning (A14)

Before candidate selection, `optimize_month` (`src/portfolio.py:optimize_month`) demeans the
signal within `gics2` for the current formation month:

$$
s_i = \text{signal}_i - \bar{\text{signal}}_{g(i)}, \qquad \bar{\text{signal}}_g = \text{mean}_{j \in g}(\text{signal}_j)
$$

(code: `s = m['signal'] - m.groupby('gics2')['signal'].transform('mean')`). Candidate selection
and the optimizer's objective both use $s$, not the raw signal.

**Why it was needed — the measured 2019-05 infeasibility.** Before this fix, candidates were the
raw top/bottom `N_CAND` by signal level. `docs/SPEC.md` §10 A14 and `docs/research_log.md`
(the Opus audit that diagnosed A14) document a concrete failure: in the 2019-05-31 validation
month, 147 of the 250 short candidates were GICS2 sector 35 and 127 of the 250 long candidates
were sector 40. With `MAX_WEIGHT = 0.015` capping how much any *other* sector's names can absorb,
the arithmetic forces sector 35's net short exposure to be at least $1 - (\text{non-35 short
candidates}) \times 0.015$ — measured at a **minimum achievable worst-sector net exposure of
9.85%**, above $2 \times$`SECTOR_TOL` $= 2\times0.03 = 6\%$, the most relaxed sector tolerance on
the `RELAX_STEPS` ladder (§4.4). No choice of weights at any ladder step could satisfy sector
neutrality — this was a *candidate-set composition* problem, not a solver bug, as the docstring on
`optimize_month` and `docs/research_log.md`'s 2026-09-27 diagnosis entry both note (that entry
independently confirms the mechanism at three real formation months, finding the worst-sector
constraint sitting exactly at its $\pm3\%$ boundary even where feasible).
`tests/test_portfolio.py:test_optimize_month_sector_lopsided_demeaning_fixes_feasibility` is the
synthetic regression analogue of this real-month failure, not its source. Across the 24 validation
months of a test year, `docs/SPEC.md` §10 records this made **3–14 of 24 validation months
infeasible**.

**Post-A14 feasibility sweep, all 92 real formation months.** A sweep over all 92 real formation
months (24 validation + 68 test) with a real-shaped mechanics signal, solved with A14's
sector-demeaning in place, passed every month at base tolerances: 0 relaxations, 0 failures,
~234 long / ~240 short names per month, and the max sector and filer exposure sitting exactly at
the `SECTOR_TOL` boundary of 0.03.

**Why the objective is (nearly) invariant to this shift.** The objective (§4.4) contains
$\text{signal}\cdot w$. Write the raw signal as its within-sector-demeaned part plus a per-sector
additive shift $c_{g}$:

$$
\text{signal}_i = s_i + c_{g(i)}, \qquad \text{signal}\cdot w = s\cdot w + \sum_g c_g \sum_{i \in g} w_i
$$

Under the sector-neutrality constraint, $\left|\sum_{i\in g} w_i\right| \le$ `SECTOR_TOL` (or its
relaxed multiple) for every sector $g$, so the second term is bounded:

$$
\left|\sum_g c_g \sum_{i\in g} w_i\right| \le \text{SECTOR\_TOL} \sum_g |c_g|
$$

— a small, roughly constant penalty independent of *which* feasible $w$ is chosen, because the
constraint caps each sector's net exposure regardless of the per-sector shift. So maximizing
$s\cdot w$ (the demeaned objective) is nearly equivalent to maximizing the raw $\text{signal}\cdot
w$ once sector neutrality is enforced anyway — the demeaning removes exactly the part of the
signal that sector neutrality would suppress from contributing to $w$'s composition, at
essentially no cost to what the optimizer is actually trying to achieve. This is the "objective
is (nearly) invariant" argument in the `optimize_month` docstring.

## 4.3 Candidate selection

Within `optimize_month`, after demeaning:

```python
long_cand  = s.nlargest(config.N_CAND).index   # N_CAND = 250
short_cand = s.nsmallest(config.N_CAND).index
names = long_cand.union(short_cand).union(w_prev.index)
```

`N_CAND = 250` per leg (`src/config.py:39`). The union with `w_prev.index` (the prior month's
held names) matters: a name held last month that is *not* a candidate this month still enters the
problem (as `is_other`), so it can be explicitly forced to zero rather than silently vanishing
(see the forced-sell constraint in §4.4).

The brief (p.14–15) describes two candidate-construction styles: decile-sorted long/short
portfolios (buy the top decile, short the bottom decile — "by construction, a dollar-neutral
portfolio whose beta still needs to be checked") or simply "buy 100 stocks for which we have the
highest predicted returns, and short sell 100 stocks for which we have the lowest." AlphaBERT
implements the second style — fixed-count top/bottom selection — with `N_CAND=250` per leg
(larger than the brief's illustrative 100, still well inside the 500-name ceiling once
`MAX_WEIGHT` capacity is accounted for, §4.4), not decile buckets.

**Sign-fixed legs.** `is_long = names.isin(long_cand)`, `is_short = names.isin(short_cand)` fix
which *leg* (long vs. short) each candidate belongs to purely by candidacy, not by the sign of its
demeaned signal $s_i$ at solve time. The optimizer then hard-constrains $w_i \ge 0$ for every
`is_long` name and $w_i \le 0$ for every `is_short` name (§4.4) — a candidate is committed to its
leg once selected, and the optimizer only chooses *how much* weight (possibly zero after
dust-rescaling, §4.4) to give it, never whether to flip it to the other leg.

## 4.4 The optimization problem

For one formation month, let $n$ be the number of names in the candidate-plus-carryover set
(`names` from §4.3). Variables: $w \in \mathbb{R}^n$, one weight per name. Known coefficients:
demeaned signal $s\in\mathbb{R}^n$, shrunk beta $\beta\in\mathbb{R}^n$ (built in
`src/data.py:_build`: $\beta_i = (1-\text{BETA\_SHRINK})\,\beta^{60m}_i + \text{BETA\_SHRINK}
\cdot 1$, `BETA_SHRINK = 0.33`, and $\beta_i=1$ if `beta_60m` is missing), size z-score $z^{sz}\in
\mathbb{R}^n$ (within-`eom` z of $\log(\text{me})$), GICS2 sector membership, `has_filing`
$\in\{0,1\}$, and the prior weight vector $w_{prev}$ (zero-filled to `names`).

**Objective** (`build_constraints`/`solve_ladder` in `optimize_month`):

$$
\max_w \;\; s^\top w \;-\; \text{TURNOVER\_PENALTY} \cdot \lVert w - w_{prev}\rVert_1 \;-\; \text{L2\_PENALTY} \cdot \lVert w \rVert_2^2
$$

- **$s^\top w$** — the forecast term: tilt weight toward names with high demeaned signal on the
  long side, low (very negative) on the short side.
- **$-\text{TURNOVER\_PENALTY}\cdot\lVert w-w_{prev}\rVert_1$** — an $\ell_1$ penalty on the
  change from last month's weights, directly discouraging trading; `TURNOVER_PENALTY` is one of
  the two calibrated penalties (§4.5). `test_turnover_penalty_reduces_turnover` confirms
  mechanically that a higher `tc` produces strictly less turnover into the same month than a
  lower one, holding everything else fixed.
- **$-\text{L2\_PENALTY}\cdot\lVert w\rVert_2^2$** — a ridge-style penalty on weight
  concentration; it pulls weights away from sitting exactly at the `MAX_WEIGHT` cap for every
  candidate and toward a more diversified book, and (with the turnover penalty) makes the
  objective strictly concave, which helps the solver find a well-conditioned unique optimum
  rather than an arbitrary vertex of a purely linear program.

**Constraints**, built in `optimize_month:build_constraints`:

1. **Leg sums (dollar / gross neutrality).**
   $$\sum_{i \in \text{long}} w_i = 1, \qquad \sum_{i \in \text{short}} w_i = -1$$
   Together these fix gross exposure $\lVert w\rVert_1 = 2$ (200% of capital: $1 long +
   1 short) and net exposure $\sum_i w_i = 0$ exactly (dollar neutral) — the brief's simplest
   neutrality definition (p.7: "the dollar value of the long book equals the dollar value of the
   short book"), and it sits at the exact centre of the brief's permitted $-50\%/+50\%$ net band
   (p.17–18) rather than merely inside it.

2. **Per-name bounds.**
   $$0 \le w_i \le \text{MAX\_WEIGHT} \;\; (i\in\text{long}), \qquad -\text{MAX\_WEIGHT} \le w_i \le 0 \;\; (i\in\text{short})$$
   `MAX_WEIGHT = 0.015` (1.5% of capital per name), enforcing diversification within each leg.

3. **Forced sells.** For every name that is neither a long nor short candidate this month
   (`is_other`, i.e. carried over from `w_prev` but no longer selected): $w_i = 0$. This is the
   mechanism that liquidates a name once it drops out of the candidate set —
   `test_noncandidate_prev_holding_is_sold` builds exactly this scenario (a held name is dropped
   from next month's universe entirely) and checks the returned weight is zero or the name is
   absent.

4. **Beta neutrality.** $\left|\beta^\top w\right| \le \text{BETA\_TOL}$, `BETA_TOL = 0.02`,
   using the *shrunk* beta from `src/data.py` (raw `beta_60m` shrunk 33% toward 1.0, missing →
   1.0) rather than a raw rolling beta — this is the brief's "harder, more honest" neutrality
   definition (p.7): a dollar-neutral book that is long high-beta and short low-beta names is
   "a levered long position wearing a disguise," and this constraint rules that out directly
   rather than relying on the ex-post regression beta (§4.6, `beta_exante`) to catch it.

5. **Sector neutrality (per GICS2).** For every sector group $g$ present among `names`:
   $$\left|\sum_{i \in g} w_i\right| \le \text{SECTOR\_TOL}, \qquad \text{SECTOR\_TOL} = 0.03$$
   This is the brief's "optional refinement" (p.7, sector/factor neutrality) — implemented as a
   hard per-sector cap on net exposure, built via `_tol_groups` (`src/portfolio.py:_tol_groups`),
   which returns one boolean mask per unique `gics2` value in the month.

6. **Size neutrality.** $\left|z^{sz\top} w\right| \le \text{SIZE\_TOL}$, `SIZE_TOL = 0.05` — a
   factor-neutrality constraint (the brief's "against size, value and momentum" example, p.7)
   preventing the book from systematically tilting small- or large-cap.

7. **Filer neutrality (A10, mandatory).** `_tol_groups` also adds a `'filer'` group
   (`has_filing == 1`) that shares `SECTOR_TOL` and the sector relaxation step:
   $$\left|\sum_{i:\ \text{has\_filing}_i=1} w_i\right| \le \text{SECTOR\_TOL}$$
   `docs/SPEC.md` §10 A10 explains why this exists relative to A1: because the text specialist's
   within-month z-score is computed among 8-K filers only (non-filers get exactly 0, per A1's
   survivorship finding — `has_filing` correlates strongly with a stock's eventual delisting, so
   it cannot be a model feature but *is* wired into how the text signal is scored), the signal
   mechanically pushes filers into the tails of the candidate ranking (measured: filer share of
   top-250 by `pred_ew` was 61% vs. a 48% base rate). Left unconstrained, the book would be
   implicitly betting on filing status itself. `test_optimize_month_filer_net_neutral` and
   `test_backtest_filer_net_neutral_diagnostic` both check `|filer_net| <= SECTOR_TOL` (with slack
   for relaxation) on signals deliberately correlated with `has_filing`.

**Solvers, status handling, relaxation ladder** (`_solve`, `RELAX_STEPS`,
`optimize_month:solve_ladder`):

`_solve(prob)` tries `CLARABEL` then `SCS` (`SOLVERS = ['CLARABEL', 'SCS']`) in order, and only
accepts `prob.status == 'optimal'` — never `'optimal_inaccurate'` or any other status — moving to
the next solver (or failing outright) otherwise. `RELAX_STEPS` is a four-rung, *cumulative*
ladder, each rung widening tolerances by a multiplier relative to the base config constants
(`sector_tol_mult, size_tol_mult, beta_tol_mult, label`):

| rung | sector_tol | size_tol | beta_tol | label |
|---|---|---|---|---|
| 1 | `SECTOR_TOL` (0.03) | `SIZE_TOL` (0.05) | `BETA_TOL` (0.02) | `''` |
| 2 | `2×SECTOR_TOL` (0.06) | `SIZE_TOL` | `BETA_TOL` | `'sector x2'` |
| 3 | `2×SECTOR_TOL` | `2×SIZE_TOL` (0.10) | `BETA_TOL` | `'sector x2,size x2'` |
| 4 | `2×SECTOR_TOL` | `2×SIZE_TOL` | `2×BETA_TOL` (0.04) | `'sector x2,size x2,beta x2'` |

`solve_ladder` tries each rung in order (rebuilding constraints with the widened tolerances,
re-solving with `_solve`) and returns the first feasible `'optimal'` result, printing what was
relaxed. This matches `docs/SPEC.md` §6's stated order exactly: "relax SECTOR_TOL, then
SIZE_TOL, then BETA_TOL by x2 steps." The leg-sum, per-name cap, and 100–500-name constraints are
never on this ladder — they are fixed competition rules (§4.4 point 1–2, and the name-count
guarantee below).

**Dust removal and rescale, then a hardened re-check.** After a feasible solve,
`_dust_and_rescale` (`src/portfolio.py:_dust_and_rescale`) zeroes any $|w_i| < \text{DUST} =
10^{-5}$ (solver noise, not a real position) and then rescales *each leg separately* back to
sum to exactly $\pm1$ (`wv[idx] *= target / wv[idx].sum()`). `_check_constraints`
(`src/portfolio.py:_check_constraints`) then asserts every constraint at `tol=1e-5` — critically,
against the **effective** tolerances actually used for that solve (`tols`, returned from
`solve_ladder`), not the fixed base config constants: `docs/research_log.md`'s 2026-09-27 entry
("`_check_constraints` relaxed-tolerance bug") records that an earlier version checked against
the fixed constants unconditionally, so *any* month that needed relaxation would spuriously fail
the post-solve assert even when its relaxed-tolerance solution was genuinely feasible. If dusting
and rescaling happen to push a constraint back out of tolerance (rescaling a leg necessarily
perturbs every other exposure on that leg slightly), `optimize_month` catches the
`AssertionError`, re-solves once more with the dusted names' weights fixed to exactly zero
(`fixed_zero=dust_mask`), dust/rescales again, and checks a second time — this second check is
*not* wrapped in a `try`, so a persistent infeasibility here propagates as a crash rather than
being silently swallowed. `test_optimize_month_forces_sector_relaxation` is the regression test
for the original bug: an engineered month is infeasible at the base tolerance by construction (a
`has_filing`-concentrated candidate set forces the filer group's net exposure strictly above
`SECTOR_TOL` but at or below $2\times$`SECTOR_TOL`) and the test asserts `relax == 'sector x2'`
and that `_check_constraints` passes at the relaxed tolerance.

**Why 100–500 names is guaranteed by construction.** With `MAX_WEIGHT = 0.015` and each leg
constrained to sum to exactly $\pm1$ with every weight bounded in magnitude by `MAX_WEIGHT`, a
leg needs at least $\lceil 1/0.015\rceil = \lceil 66.67\rceil = 67$ nonzero names to reach a sum
of 1 in absolute value. So **both legs together need at least 134 names**, comfortably above the
brief's 100-name floor (p.17) without the assert doing any real work in that direction. On the
other side, each leg's nonzero names can only come from that leg's $N_{\text{CAND}}=250$
candidates (non-candidates are forced to zero, point 3 above), so the total can never exceed
$2\times250=500$ — exactly the brief's ceiling. `_check_constraints`'s final assert,
`assert 100 <= n_names <= 500`, is therefore a sanity check on an outcome that the cap arithmetic
already guarantees, not something that could plausibly fail from a correctly wired optimizer.

## 4.5 Calibration (A3, amended)

`calibrate()` (`src/portfolio.py:calibrate`) grid-searches the two penalties by literally rerunning
`backtest()` (§4.6) at each grid point — so calibration exercises the exact same optimizer path,
including the mandatory filer-net constraint, as the real backtest (`docs/SPEC.md` §10 A3
amendment: "calibrate() now reuses backtest()'s per-month optimizer call directly").

- Grid: `l2_grid = [100.0, 300.0, 1000.0, 3000.0]`, `tc_grid = [0.1, 0.3, 1.0, 3.0]` — 16 cells
  (`tests/test_integrity.py` asserts `len(calib_table) == 16`).
- Objective (shape only, never returns):
  $$
  \text{score}(l_2, tc) = \left|\overline{n} - 150\right| + \left|\overline{\tau} - 0.3\right| \times 150
  $$
  where $\overline{n} = \text{mean}\!\left(\frac{n_{\text{long}}+n_{\text{short}}}{2}\right)$
  (average names per side, target `target_names_per_side=150`) and $\overline{\tau}$ is mean
  one-way turnover over all *non-first* months (target `target_turnover=0.3`). The
  turnover term is scaled by 150 so a 0.1 miss in turnover ($\approx$ the scale of a 15-name miss)
  is weighted comparably to a 15-name miss in $\overline{n}$ — both terms are on a
  roughly-comparable "names" scale. The best `(l2, tc)` (`table.loc[table['score'].idxmin()]`) is
  returned along with the full 16-row table.
- **Only 2019–2020 validation signal, never returns.** `MAIN.py:main` (step 4) builds this signal
  from `preds.loc[preds['split']=='valid', ['permno','eom','pred_ew']]` — the un-fitted,
  equal-weight blend `pred_ew` (no combiner), smoothed via `portfolio.smooth` per the A3
  amendment (previously the raw within-`eom` z-score). `calibrate()`'s own scoring function only
  ever touches `n_long`, `n_short`, and `turnover` from the `returns` frame `backtest()` produces
  — it never reads `stock_exret`-derived columns (`ls_ret`, `total_ret`, etc.), even though
  `backtest()` computes them as a byproduct of being reused wholesale. This is what "decided
  without seeing any test-period returns" (`docs/SPEC.md` §10 A3/A14) means operationally.
- **Locked afterwards.** `MAIN.py` calls `calibrate()` exactly once (step 4) and threads the
  returned `l2, tc` into every subsequent `backtest()` call — the headline (step 5) and every
  ablation (step 6) — via `run_signal(col, test_preds, panel, market, l2, tc)`. `optimize_month`'s
  own `l2=None, tc=None` defaults fall back to `config.L2_PENALTY` / `config.TURNOVER_PENALTY`
  (100.0 / 0.5) only when no explicit value is passed (e.g. in unit tests that don't calibrate);
  those config constants are explicitly commented `# placeholder, calibrated later on 2019-2020
  validation only` (`src/config.py:42-43`) and are **not** claimed anywhere to equal the value
  `calibrate()` would lock in — that locked value is only produced by actually running
  `MAIN.py`, which (per this task's constraints) has not happened yet.

## 4.6 Backtest accounting

`backtest()` (`src/portfolio.py:backtest`) loops over sorted formation months, calling
`optimize_month` once per month and `compute_month_return` (`src/portfolio.py:compute_month_return`)
to turn the resulting weights into a row of the `returns` table. This section derives every
column algebraically from the code.

**Capital convention.** Per `docs/SPEC.md` §6: $100 of investor capital, $100 deployed long
(leg sums to $+1$, i.e. +100% of capital) and $100 deployed short (leg sums to $-1$); the
uninvested collateral behind the position earns the risk-free rate `rf_m`. `stock_exret`
(`ret_exc_lead1m`) is *already* an excess return — raw return minus whatever risk-free series the
raw data's own `ret_exc`/`ret_exc_lead1m` construction used (call it $rf^{\text{pipeline}}$,
which the brief explicitly warns (p.8) need not be the same series as the competition's own
`TB3MS`-based `rf_m`).

**Deriving `ls_ret` and why the pipeline RF cancels.** For a name $i$ with weight $w_i$ (as a
fraction of the $100 capital) and excess return $r_i = \text{stock\_exret}_i$, its raw total
return is $r_i^{\text{raw}} = r_i + rf^{\text{pipeline}}$. The dollar P&L on the whole book is

$$
\Delta\$ = 100\sum_{i} w_i\, r_i^{\text{raw}} = 100\sum_i w_i\, r_i \;+\; 100\, rf^{\text{pipeline}} \sum_i w_i
$$

Because the long leg sums to $+1$ and the short leg sums to $-1$, $\sum_i w_i = 0$ **exactly**
(constraint 1, §4.4), so the second term vanishes regardless of what $rf^{\text{pipeline}}$
actually is. What remains, per $1 of capital, is exactly what the code computes:

$$
\texttt{long\_ret} = \sum_{i:\,w_i>0} w_i r_i, \quad \texttt{short\_ret} = \sum_{i:\,w_i<0} w_i r_i, \quad \texttt{ls\_ret} = \texttt{long\_ret} + \texttt{short\_ret}
$$

(code: `long_ret = float((weights[long_mask] * r[long_mask]).sum())`, similarly for `short_ret`).
This is the "pipeline RF cancels" claim in `docs/SPEC.md` §6 verified algebraically: `ls_ret` is a
clean long-short spread return with no dependence on which risk-free series the raw panel used to
build `stock_exret`.

**Total, benchmark, active return.** The competition's own risk-free rate (`rf_m`, from
`market.loc[holding_month]`, itself `TB3MS/1200` per `docs/SPEC.md` §3) is then added back as the
collateral return:

$$
\texttt{total\_ret} = rf_m + \texttt{ls\_ret}, \qquad \texttt{bench\_ret} = rf_m + \frac{\text{HURDLE\_ANNUAL}}{12} = rf_m + \frac{0.04}{12}
$$

$$
\texttt{active\_ret} = \texttt{total\_ret} - \texttt{bench\_ret} = \texttt{ls\_ret} - \frac{0.04}{12}
$$

— `rf_m` cancels a second time in `active_ret`, so the strategy's active return depends only on
the stock-selection spread `ls_ret` and the fixed 4%/year hurdle, exactly matching the brief's
formula (p.8): $\text{active}_t = R_{\text{portfolio,total},t} - (TB3MS_t/100/12 + 0.04/12)$.
`bench_ret`/`active_ret` use the *same* `rf_m` on both sides of the subtraction, so this identity
holds by construction, not by any numerical coincidence.

**`long_ret`/`short_ret` contributions** are the two summands of `ls_ret` above — each leg's
raw contribution to the spread, useful for attributing performance between stock-picking on the
long side vs. the short side (used downstream by `evaluate.py`'s regime table).

**Turnover.** Target-to-target, i.e. against the *previous month's target weights* `w_prev`, not
any intra-month drift:

$$
\texttt{turnover} = \frac{0.5\sum_i |w_i - w_{prev,i}|}{\text{GROSS}}, \qquad \text{GROSS}=2.0
$$

computed over the union of this month's and last month's names (`dw = weights.reindex(all_names)
- w_prev.reindex(all_names)`, both zero-filled). One-way, as a fraction of gross: a full
replacement of both legs (every name in $w$ different from every name in $w_{prev}$) gives
$\sum|w_i-w_{prev,i}| = 4$ (2 from the long leg turning over completely, 2 from the short leg),
so $\texttt{turnover} = 0.5\times4/2 = 1.0$ — "full replacement = 1.0" as `docs/SPEC.md` §6
states. **The first month** starts from an empty `w_prev` (`pd.Series(dtype=float)` in
`backtest`), so `turnover` there only reflects "buying from nothing" rather than a genuine
round-trip and is not comparable to later months; `backtest` flags it via `rec['first_month'] =
(i == 0)` and `docs/SPEC.md` §6 states it is excluded from averages downstream (by `evaluate.py`,
outside this chapter's file).

**Cost.** $\texttt{cost} = \frac{\text{COST\_BPS}}{10^4}\sum_i|w_i - w_{prev,i}|$, `COST_BPS =
10` (10bps per unit of $\sum|\Delta w|$, i.e. per dollar of gross traded as a fraction of
capital — note this is *not* divided by `GROSS`/2 the way `turnover` is, so `cost` scales with
raw two-way traded notional, not the turnover ratio). Net-of-cost returns are then
$\texttt{total\_ret\_net} = \texttt{total\_ret} - \texttt{cost}$ and
$\texttt{active\_ret\_net} = \texttt{active\_ret} - \texttt{cost}$ — gross figures remain the
reported headline (`docs/SPEC.md` §10 A5), net figures are companions.

**Missing next-month returns.** `r = ret.reindex(idx); missing = r.isna(); r = r.fillna(0.0)` —
a name with no realized `stock_exret` this holding month (a true exit, per A4) contributes
exactly 0 to `long_ret`/`short_ret` rather than being dropped or imputed with a nonzero value.
`missing_ret_weight = weights[missing].abs().sum()` records how much absolute weight was
affected, so the headline's silent-zero treatment is auditable per month.

**Adverse sensitivity.** `missing_return_sensitivity()` (`src/portfolio.py:missing_return_sensitivity`)
is a *separate*, side-only check — never used to fill training labels. For every holding whose
`stock_exret` was missing (joined back from `panel`), it adds `weight * fill` where `fill =
-0.30` if the position was long (`weight > 0`) or `+0.30` if short (`weight < 0`) — i.e. it
assumes the worst plausible re-pricing on exit (longs fall 30%, shorts rise 30%, both adverse to
the position actually held), matching `docs/SPEC.md` §10 A4's "(-30% longs / +30% shorts)"
exactly. Only `long_ret`, `short_ret`, `ls_ret`, `total_ret`, `active_ret` (and their `_net`
companions) are recomputed from these adjustments; `turnover`, `cost`, `gross`, `net` depend only
on weights (unaffected by which names had a realized return), so the function copies them through
unchanged — `test_integrity.py`'s mini pipeline test checks this explicitly
(`pd.testing.assert_series_equal(sensitivity[col], returns[col])` for those four columns).

**`filer_net`, `relax`, `beta_exante`.**
- `rec['filer_net'] = (w * has_filing).sum()` — computed in `backtest`, the ex-post realized
  filer exposure; constrained ex-ante to $\le\text{SECTOR\_TOL}$ (possibly relaxed) by A10
  (§4.4 point 7).
- `rec['relax']` — the (possibly comma-joined) label(s) from `solve_ladder`, e.g. `''`,
  `'sector x2'`, or `'sector x2,sector x2'` if a second relaxation was needed on the dust-recheck
  re-solve (§4.4). `tests/test_integrity.py`'s mini end-to-end test checks this column is present.
- `beta_exante = (weights * beta_map).sum()` — the ex-ante, model-implied beta exposure of the
  book (constrained $\le\text{BETA\_TOL}$ by construction, §4.4 point 4), distinct from the
  ex-post *realized* beta the brief asks to be estimated by OLS regression on actual monthly
  returns (p.15) — that regression lives in `evaluate.py`, outside this chapter, but
  `beta_exante` is the per-month sanity check that the constrained optimizer actually delivered a
  near-zero-beta book going in.

## 4.7 Labels

**`attach_labels()` fallback chain** (`src/portfolio.py:attach_labels`, helper `_asof_fill`):
holdings already carry a `label_source` of `'panel'` where `backtest()` found a non-null
`ticker`/`company_name` directly on the formation-month panel row (`src/portfolio.py:backtest`,
`h['label_source'] = np.where(h['ticker'].notna() & h['company_name'].notna(), 'panel', None)`).
For everything still missing, `attach_labels` tries, in order:

1. **Same-month panel first, then earlier panel.** `_asof_fill` against `label_panel` (the raw
   characteristics file's `ticker`/`company_name` history) using `pd.merge_asof(..., on='eom',
   by='permno', direction='backward')` — for each missing `(permno, eom)`, the most recent
   labelled row of that permno with `eom <= (\text{holding's }eom)$. Tag: `'raw_panel'`.
2. **Earlier 8-K filing.** Same `_asof_fill` mechanism against `filing_labels`, matched on
   `filing_date <= eom` instead. Tag: `'filing'`.
3. **`'UNLABELED'`** if nothing before or at `eom` exists in either source.

`direction='backward'` in `merge_asof` is what guarantees **never a later label**: it only ever
matches a source row whose date is $\le$ the holding's `eom`, by construction of `merge_asof`'s
backward mode — there is no code path that could select a future-dated ticker/name.
`test_attach_labels_never_uses_future_dates` verifies this directly: a label_panel row dated
*after* `eom` for permno 1, and a filing_labels row dated after `eom` for permno 2, are both
present in the fixture but never selected — permno 1 gets the earlier `'OLD'`/`raw_panel` label,
permno 2 gets the earlier `'F2'`/`filing` label, and permno 3 (no history at all) becomes
`'UNLABELED'`.

**`write_submission()`** (`src/portfolio.py:write_submission`) produces three files into
`config.SUB_DIR`:

- **`holdings.csv`**: `Date` = first day of the *holding* month
  (`h['month'].dt.to_period('M').dt.to_timestamp()`, formatted `%Y-%m-%d`) — matches the brief's
  own example "Sept 01 2026" (p.22) and `docs/SPEC.md` §10 A5's "first-of-holding-month Date like
  holdings.csv." Columns: `PERMNO`, `TICKER`, `COMPANY NAME`, `WEIGHT` (percent of NAV, signed:
  e.g. `1.5` means long 1.5%, `-1.5` means short 1.5%, matching the brief's "Positive for Long and
  Negative for Short Positions," p.22). Rounding, via `_round_submission_weights`
  (`src/portfolio.py:_round_submission_weights`): each leg is rounded to 6 decimal places, and any
  residual from rounding is placed entirely on the *largest-|weight| name of that leg* — but only
  if adding the residual keeps that name within the `MAX_WEIGHT`-derived cap (`cap_pct =
  MAX_WEIGHT*100 = 1.5`); otherwise the leg is left with its small rounding residual rather than
  breach the cap. After rounding, `write_submission` asserts every month's long leg sums to
  exactly $+100.000000$ and short leg to exactly $-100.000000$ (`abs(g[g>0].sum()-100.0) <
  1e-6`), and that no `WEIGHT` exceeds the cap.
- **`returns.csv`**: `Date` (same first-of-holding-month convention), then `total_ret, rf_m,
  bench_ret, active_ret, ls_ret, long_ret, short_ret, sp500_ret, total_ret_net, active_ret_net`,
  in decimal units — the exact column list and order from `docs/SPEC.md` §10 A5.
- **`label_audit.csv`**: `permno, month (first-of-holding-month string), ticker, company_name,
  label_source` — the per-holding audit trail feeding A8's deck-visible-holding EDGAR
  verification (outside this chapter's scope).

`write_submission` also hard-asserts `h['ticker'].notna().all() and
h['company_name'].notna().all()` before writing anything — holdings must have already passed
through `attach_labels` (which guarantees no nulls, falling back to the literal string
`'UNLABELED'`); `test_write_submission_rejects_null_labels` confirms a null `ticker` raises
`AssertionError` rather than silently writing a blank cell.

## 4.8 A worked numerical example

This is a hand-built pedagogical toy with invented permnos and returns — not a real backtest
output (none has been run). It illustrates one month's accounting mechanics end to end, using 3
long and 3 short names. Weights like 0.5 are scaled up here purely for readability; real weights
are capped at `MAX_WEIGHT = 1.5%` per name, so a real leg holds at least 67 names (§4.4).

**Setup.** Current-month target weights (already satisfying $\sum_{\text{long}}w_i=1$,
$\sum_{\text{short}}w_i=-1$, and, with the betas below, $|\beta^\top w|\le\text{BETA\_TOL}$):

| name | leg | $w$ | $r=$`stock_exret` | $\beta$ |
|---|---|---|---|---|
| A | long | 0.5 | 0.04 | 1.05 |
| B | long | 0.3 | -0.02 | 0.95 |
| C | long | 0.2 | 0.01 | 1.00 |
| D | short | -0.5 | 0.03 | 1.00 |
| E | short | -0.3 | -0.01 | 1.00 |
| F | short | -0.2 | *missing* | 1.00 |

Prior month's weights $w_{prev}$: A 0.4, B 0.3, G 0.3 (long); D -0.4, E -0.3, H -0.3 (short) — G
and H are no longer candidates this month (forced-sold, §4.4 point 3), and C/F are new. Market
data for the holding month: $rf_m = 0.0025$ (0.25%/month).

**Long/short/ls_ret** (`compute_month_return`, §4.6). F's return is missing, so it is 0-filled:

$$
\texttt{long\_ret} = 0.5(0.04)+0.3(-0.02)+0.2(0.01) = 0.020-0.006+0.002 = 0.016
$$
$$
\texttt{short\_ret} = -0.5(0.03)+(-0.3)(-0.01)+(-0.2)(0) = -0.015+0.003+0 = -0.012
$$
$$
\texttt{ls\_ret} = 0.016 + (-0.012) = 0.004, \qquad \texttt{missing\_ret\_weight} = |{-0.2}| = 0.2
$$

**Total / bench / active:**

$$
\texttt{total\_ret} = 0.0025 + 0.004 = 0.0065
$$
$$
\texttt{bench\_ret} = 0.0025 + 0.04/12 = 0.0025+0.003333=0.005833
$$
$$
\texttt{active\_ret} = 0.0065-0.005833 = 0.000667 \;\;(=\; \texttt{ls\_ret}-0.04/12 = 0.004-0.003333)
$$

**Turnover and cost.** Union of names: {A,B,C,D,E,F,G,H}. $\Delta w$: A: $0.5-0.4=0.1$; B:
$0.3-0.3=0$; C: $0.2-0=0.2$; D: $-0.5-(-0.4)=-0.1$; E: $-0.3-(-0.3)=0$; F: $-0.2-0=-0.2$; G:
$0-0.3=-0.3$; H: $0-(-0.3)=0.3$. $\sum|\Delta w| = 0.1+0.2+0.1+0.2+0.3+0.3 = 1.2$.

$$
\texttt{turnover} = \frac{0.5\times1.2}{2.0} = 0.30, \qquad \texttt{cost} = \frac{10}{10^4}\times1.2 = 0.0012
$$
$$
\texttt{total\_ret\_net} = 0.0065-0.0012=0.0053, \qquad \texttt{active\_ret\_net}=0.000667-0.0012=-0.000533
$$

(Turnover lands at exactly the calibration target of $\approx0.3$ from §4.5 by deliberate choice
of this toy's numbers — a pedagogical coincidence, not a claim about the real calibrated
penalties.)

**Beta exposure:**

$$
\beta^\top w = 0.5(1.05)+0.3(0.95)+0.2(1.00) - \big[0.5(1.00)+0.3(1.00)+0.2(1.00)\big] = 1.010-1.000=0.010
$$

$|0.010|\le\text{BETA\_TOL}=0.02$ — within tolerance.

**Gross / net:** $\texttt{gross}=\sum|w_i| = (0.5+0.3+0.2)+(0.5+0.3+0.2)=2.0$;
$\texttt{net}=\sum w_i = 1.0-1.0=0$.

## 4.9 Tests and what they prove

`tests/test_portfolio.py` (unit-level, synthetic data, fast):

- **`test_smooth_past_only_and_reset`** — hand-computed EMA chain confirms the $\alpha=0.5$
  recursion and the gap-reset rule; mutating a later month's input leaves earlier smoothed values
  unchanged (past-only, §4.1).
- **`test_optimize_month_constraints`** (`_check_month` helper, run across 6 sequential months) —
  every constraint from §4.4 (leg sums, cap, beta/size/sector/filer tolerances, 100–500 names)
  holds on ordinary random synthetic data, chained month to month with `w_prev`.
- **`test_turnover_penalty_reduces_turnover`** — a higher `tc` produces strictly less turnover
  into the same target month, proving the turnover term functions as intended.
- **`test_optimize_month_filer_net_neutral`** — signal correlated (not deterministic) with
  `has_filing` still yields $|\texttt{filer\_net}|\le\text{SECTOR\_TOL}$ (A10, §4.4 point 7).
- **`test_optimize_month_forces_sector_relaxation`** — the regression test for the
  `_check_constraints` relaxed-tolerance bug (§4.4): an engineered month is infeasible at the
  base tolerance by construction (forced filer-group exposure strictly between `SECTOR_TOL` and
  $2\times$`SECTOR_TOL`), and the test asserts the solver lands on exactly `relax == 'sector x2'`
  and that the *relaxed* tolerance holds.
- **`test_optimize_month_sector_lopsided_demeaning_fixes_feasibility`** — the regression test for
  A14 (§4.2): a synthetic month with two sectors carrying a large additive signal shift (which
  would make the raw top/bottom-N_CAND selection infeasible even after full relaxation, verified
  by an assertion on the *raw*-signal candidate composition inside the test itself) becomes
  feasible once `optimize_month`'s within-sector demeaning removes the shift.
- **`test_noncandidate_prev_holding_is_sold`** — a name held last month that is entirely absent
  from this month's universe is force-sold to zero (§4.4 point 3).
- **`test_compute_month_return_hand_computed`** — every field of `compute_month_return`'s output
  (long/short/ls/total/bench/active return, missing_ret_weight, gross, net, turnover, cost, net
  returns, beta_exante) checked against hand arithmetic on a 5-name toy — essentially §4.8's
  formulas verified in code rather than by hand.
- **`test_missing_return_sensitivity_hand_computed`** — the adverse $-30\%/+30\%$ adjustment
  (§4.6) checked by hand on a 4-name toy with two missing returns, one long and one short.
- **`test_backtest_tiny_synthetic`** — proves `backtest` indexes `market` by the *holding* month
  ($eom+1$ month-end), not the formation month — a distinct, distinguishable `rf_m`/`sp500_ret`
  per holding month would only match if the lookup used the correct date.
- **`test_backtest_filer_net_neutral_diagnostic`** — same filer-net check as above, run through
  the full `backtest()` loop rather than a single `optimize_month` call.
- **`test_backtest_raises_if_holding_month_missing_from_market`** — `backtest` raises `ValueError`
  rather than silently producing a `NaN` `rf_m` when the holding month isn't in `market`.
- **`test_attach_labels_never_uses_future_dates`** — §4.7's backward-only `merge_asof` guarantee,
  checked with deliberately-planted future-dated rows in both label sources.
- **`test_write_submission_format_and_rounding`** / **`test_write_submission_rejects_null_labels`**
  — the three CSVs' columns, `Date` formatting, weight rounding to exactly $\pm100$ per leg, the
  `MAX_WEIGHT` cap, and the null-label guard (§4.7).

`tests/test_integrity.py` (slower, closer to the real pipeline):

- **`test_mini_end_to_end_pipeline`** (marked `@pytest.mark.slow`) — runs the actual
  `models.run_all -> smooth -> backtest -> attach_labels -> write_submission ->
  evaluate.run_evaluation -> calibrate -> missing_return_sensitivity` chain on a 600-permno
  subset for test_year 2021, proving the interfaces compose (not reporting any performance
  number). It explicitly re-checks the 100–500 name count, leg sums, `MAX_WEIGHT` cap, the full
  `returns` column set (including `filer_net`, `relax`, `first_month`), a relaxation-aware filer
  tolerance check (`2×SECTOR_TOL` where `relax` mentions `'sector'`, `SECTOR_TOL` otherwise),
  `calibrate()`'s 16-row grid and positive `l2`/`tc`, and that `missing_return_sensitivity`
  leaves `turnover`/`cost`/`gross`/`net` byte-identical to `backtest`'s own output. Its docstring
  notes this test was previously `xfail` pending A14 and now asserts a real pass — direct
  evidence the sector-demeaning fix (§4.2) resolved the infeasibility on realistic (if
  small-sample) data, not just the hand-engineered unit tests.

---

### A note on documentation lag (not a code bug)

`docs/SPEC.md` §6's prose description of `optimize_month` predates amendments A10 (filer
neutrality) and A14 (sector-demeaned candidate selection) and does not mention either; both are
implemented in the code and documented correctly in §10. A reader consulting §6 alone would get
an incomplete picture of the current constraint set — §10 is the source of truth for A10/A14.
Separately, `config.L2_PENALTY`/`config.TURNOVER_PENALTY` (100.0 / 0.5) are explicitly commented
as placeholders (`src/config.py:42-43`); the actual locked values only exist once `MAIN.py`'s
`calibrate()` step has been run, which it has not been as of this chapter.
