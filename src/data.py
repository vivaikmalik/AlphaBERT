"""Panel construction, universe, ranks, market state, external market data. See docs/SPEC.md section 3."""
import hashlib
import io
import urllib.request

import numpy as np
import pandas as pd

from src import config
from src import beta as beta_mod
from src.beta import compute_betas

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
