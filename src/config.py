import os

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
