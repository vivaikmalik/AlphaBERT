"""Panel construction, universe, ranks, market state, external market data. See docs/SPEC.md section 3."""
import io
import urllib.request

import numpy as np
import pandas as pd

from src import config

FRED_TB3MS_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS'
FRED_SP500_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500'


def load_chars(columns=None) -> pd.DataFrame:
    df = pd.read_parquet(config.CHARS_PATH, columns=columns)
    for c in ('eom', 'date'):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).astype('datetime64[ns]')
    return df


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
    beta_raw = df['beta_60m'].copy()

    # miss_ flags: selected on universe rows with eom <= cutoff, applied to all universe rows
    cutoff_rows = df.loc[df['eom'] <= config.MISS_FLAG_CUTOFF, chars]
    miss_rate = cutoff_rows.isna().mean()
    miss_chars = miss_rate[miss_rate > config.MISS_FLAG_RATE].index.tolist()
    print(f'miss_ flags selected ({len(miss_chars)}): {miss_chars}')
    miss_flags = {f'miss_{c}': df[c].isna().astype('int8') for c in miss_chars}

    scaled = _rank_within_eom(df, chars)

    # aux (not features)
    gics2 = pd.Series(np.where(df['gics'].isna(), 'NA', df['gics'].astype(str).str.slice(0, 2)), index=df.index)
    beta = pd.Series(
        np.where(beta_raw.isna(), 1.0, (1 - config.BETA_SHRINK) * beta_raw + config.BETA_SHRINK * 1.0),
        index=df.index)
    log_me = np.log(df['me'])
    size_z = log_me.groupby(df['eom']).transform(lambda s: (s - s.mean()) / s.std())
    # NOTE deviation from literal SPEC wording: 'prc' and 'dolvol_126d' are both feature-char names
    # (ranked above) and requested as raw aux columns under the same name, which collide. We keep
    # the ranked feature under the standard name (needed by models.py) and expose the raw value
    # under '<name>_raw' instead. 'me' has no such collision and stays raw.
    aux = pd.DataFrame({
        **miss_flags,
        'gics2': gics2, 'beta': beta, 'size_z': size_z,
        'prc_raw': prc_raw, 'dolvol_126d_raw': dolvol_raw,
    }, index=df.index)
    lead = pd.DataFrame({'target_month': target_month, 'stock_exret': stock_exret}, index=df.index)

    panel = pd.concat([df[['permno', 'eom']], lead, df[['date']], scaled, aux,
                        df[['me', 'size_grp', 'ticker', 'company_name']]], axis=1)
    return panel


def build_panel() -> pd.DataFrame:
    cache_path = config.CACHE_DIR / 'panel.parquet'
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

    aug2026 = pd.Timestamp('2026-08-31')
    if aug2026 not in df.index:
        print('WARNING: load_market has no row for 2026-08-31 (TB3MS and/or S&P 500 not yet published).')
    elif df.loc[[aug2026]].isna().any().any():
        print('WARNING: load_market 2026-08-31 row contains NaN values.')
    return df
