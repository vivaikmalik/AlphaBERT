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
BETA_SHRINK = 0.33

KEY_ITEMS = ['1.01', '1.02', '2.01', '2.02', '2.05', '2.06', '3.01', '4.01', '4.02', '5.02', '7.01', '8.01']
STATE_VARS = ['mkt_vol12', 'disp']

GROSS = 2.0
MAX_WEIGHT = 0.015
N_CAND = 250
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
