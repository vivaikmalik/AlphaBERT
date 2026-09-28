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
BETA_SHRINK = 0.33  # unused by src/data.py as of A15 (beta model fix); kept because docs/guide/*.md
# and docs/SPEC.md section 3 (not owned by this amendment -- only section 10 may be appended)
# still describe the pre-A15 beta formula in terms of it.

# A15 (beta model fix, docs/SPEC.md section 10): beta = BETA_INTERCEPT + BETA_SLOPE*b1 + BETA_IVOL*ivp,
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
N_CAND = 350  # A15: raised from 250 -- validation beta was too short-lopsided at 250 (see A15 log)
EMA_ALPHA = 0.5

TURNOVER_PENALTY = 0.5   # placeholder, calibrated later on 2019-2020 validation only
L2_PENALTY = 100.0       # placeholder, calibrated later on 2019-2020 validation only

BETA_TOL = 0.02
SECTOR_TOL = 0.03
SIZE_TOL = 0.05

COST_BPS = 10
HURDLE_ANNUAL = 0.04

N_JOBS = 6


def load_char_list() -> list[str]:
    df = pd.read_csv(CHAR_LIST_PATH)
    names = df['variable'].tolist()
    assert len(names) == 147
    return names
