"""8-K text cleaning, FinBERT scoring (cached), and text features. See docs/SPEC.md section 4.

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
"""
import argparse
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import config

FINBERT_MODEL = 'ProsusAI/finbert'
FINBERT_REVISION = '4556d13015211d73dccd3fdd39d39232506f3e43'  # pinned commit (HF hub main @ 2026-09-27)
NUM_THREADS = 10
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
    from src import data

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
    elif args.score:
        score_finbert(max_length=args.max_length, device=args.device, batch_size=args.batch_size)
    else:
        print('usage: python -m src.text --score [--device {auto,cpu,cuda}] [--max-length N] [--batch-size N]')
        print('       python -m src.text --check [N] [--device {auto,cpu,cuda}]')
