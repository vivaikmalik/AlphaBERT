"""Unit tests for src/text.py. Fast; FinBERT is mocked (never loaded here)."""
import numpy as np
import pandas as pd
import pytest

from src import config, text


# ---------------------------------------------------------------- clean_text: cover page / item title

def test_clean_text_drops_cover_page_and_masks_names():
    cover = (
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 8-K "
        "CURRENT REPORT ACME WIDGETS CORP (Exact Name of Registrant as Specified in its Charter) "
        "Delaware 001-12345 12-3456789 Check the appropriate box below if the Form 8-K filing is "
        "intended to satisfy... Indicate by check mark whether the registrant is an emerging "
        "growth company as defined in Rule 405. Emerging growth company [ ] "
    )
    body = (
        "Item 8.01 Other Events. On June 1, 2024, Acme Widgets Corp (the \"Company\") announced "
        "that Acme will expand its plant, trading on the Nasdaq under the ticker symbol \"ACME\". "
        "ACME shares last closed up 3%. This does not mangle the word company in lowercase or the "
        "word acmen."
    )
    names = {'company_name': 'ACME WIDGETS CORP', 'content_company_name': 'Acme Widgets Corp', 'ticker': 'ACME'}
    out = text.clean_text(cover + body, names)

    assert 'SECURITIES AND EXCHANGE COMMISSION' not in out
    assert 'emerging growth company' not in out
    assert 'Item 8.01' not in out  # boilerplate item title skipped; body leads with the event
    assert 'Acme Widgets Corp' not in out  # full name, case-insensitive
    assert 'ticker symbol "ACME"' not in out  # ticker masked: exchange context present
    assert 'ACME shares last closed up 3%' in out  # bare ticker mention (no exchange context) untouched
    assert out.count('the Company') == 2  # the "Acme Widgets Corp" phrase, and the in-context ticker
    # normal words containing the same letters must survive untouched: masking only replaces the
    # full name/ticker variants, not arbitrary substrings of ordinary words
    assert 'the word company in lowercase' in out
    assert 'acmen' in out
    assert 'that Acme will expand' in out  # bare "Acme" (not a qualifying short variant) untouched


def test_clean_text_ticker_masked_only_in_exchange_context():
    names = {'company_name': None, 'content_company_name': None, 'ticker': 'MA'}
    out = text.clean_text(
        'Item 8.01 Other Events. The Company trades on the NYSE: MA. Some contracts require MA '
        'state approval, unrelated to the ticker.',
        names,
    )
    assert 'NYSE: MA' not in out
    assert 'NYSE: the Company' in out
    assert 'MA state approval' in out  # bare ticker mention (no exchange marker before it) untouched


def test_clean_text_short_name_variant_needs_two_words_or_six_chars():
    names = {'company_name': 'Box, Inc.', 'content_company_name': None, 'ticker': None}
    out = text.clean_text(
        'Item 1.01 Entry. Box, Inc. entered into an agreement. Please box the equipment before '
        'shipping; the box was returned.',
        names,
    )
    assert 'Box, Inc.' not in out  # full legal name still masked (case-insensitive)
    assert 'the Company entered into an agreement' in out
    assert 'Please box the equipment' in out  # short "Box" (3 chars, 1 word) not masked as a bare word
    assert 'the box was returned' in out


def test_clean_text_short_name_variant_masked_when_long_or_multiword():
    names = {'company_name': 'Target Corporation', 'content_company_name': None, 'ticker': None}
    out = text.clean_text(
        'Item 1.01 Entry. Target Corporation entered a deal. Target announced results today.',
        names,
    )
    assert 'Target Corporation' not in out  # full name
    assert 'the Company announced results today' in out  # bare "Target" (6 chars) masked, case-sensitive


def test_clean_text_short_variant_is_case_sensitive():
    # suffix present so the full-name variant ("Acme Widgets Corp") doesn't also match the bare
    # lowercase mention below; only the stripped short variant ("Acme Widgets") is at play here.
    names = {'company_name': 'Acme Widgets Corp', 'content_company_name': None, 'ticker': None}
    out = text.clean_text(
        'Item 1.01 Entry. Acme Widgets entered a deal. some acme widgets brand mugs were sold too.',
        names,
    )
    assert 'the Company entered a deal' in out  # exact-case "Acme Widgets" (short variant) masked
    assert 'acme widgets brand mugs' in out  # lowercase variant untouched (case-sensitive match only)


def test_clean_text_skips_boilerplate_item_title_when_on_sentence_found():
    names = {'company_name': None, 'content_company_name': None, 'ticker': None}
    out = text.clean_text('Item 8.01 Other Events. On June 1, 2024, something happened.', names)
    assert out == 'On June 1, 2024, something happened.'


def test_clean_text_keeps_title_when_no_on_sentence_nearby():
    names = {'company_name': None, 'content_company_name': None, 'ticker': None}
    out = text.clean_text('Item 5.02 Departure of Directors. (c) Jane Doe resigned today.', names)
    assert out.startswith('Item 5.02')  # safe fallback: no "On " found, title is kept


def test_clean_text_skips_table_of_contents_item_mention():
    # regression: a real 8-K's "TABLE OF CONTENTS" block re-lists the item number/title before
    # the real section header repeats it; the first _ITEM_RE match must not land on the TOC
    # entry (which is followed only by navigation text: SIGNATURES, EXHIBIT INDEX, ...), or the
    # whole real body gets discarded.
    names = {'company_name': None, 'content_company_name': None, 'ticker': None}
    raw = (
        'UNITED STATES SECURITIES AND EXCHANGE COMMISSION FORM 8-K CURRENT REPORT '
        'TABLE OF CONTENTS Item 5.02 Departure of Directors. Financial Statements and Exhibits '
        'SIGNATURES EXHIBIT INDEX '
        'Item 5.02 Departure of Directors. On January 1, 2015, the Company entered into '
        'employment agreements with its executives.'
    )
    out = text.clean_text(raw, names)
    assert 'entered into employment agreements' in out
    assert 'Financial Statements and Exhibits SIGNATURES EXHIBIT INDEX' not in out


def test_clean_text_tolerates_malformed_item_header_spacing():
    names = {'company_name': None, 'content_company_name': None, 'ticker': None}
    out = text.clean_text('Item 5.0 7 Departure of Directors. (c) Jane Doe resigned today.', names)
    assert out.startswith('Item 5.0 7')  # matched despite the odd digit spacing; kept (no "On " nearby)


def test_clean_text_fallback_when_no_item_pattern():
    text_no_item = 'X' * 2000
    out = text.clean_text(text_no_item, {'company_name': None, 'content_company_name': None, 'ticker': None})
    assert len(out) == 500  # dropped first 1500 chars


def test_clean_text_short_text_no_item_pattern_keeps_whole_text():
    # a naive first-1500-chars drop would empty this text; too short to assume it's a cover page
    short_text = 'X' * 500
    out = text.clean_text(short_text, {'company_name': None, 'content_company_name': None, 'ticker': None})
    assert out == short_text


# ---------------------------------------------------------------- clean_text: boilerplate stripping

def test_clean_text_strips_furnished_safe_harbor_sentence():
    names = {'company_name': None, 'content_company_name': None, 'ticker': None}
    raw = (
        'Item 2.02 Results of Operations and Financial Condition. On March 3, 2025, the Company '
        'issued a press release announcing its financial results for the quarter. The information '
        'in this Item 2.02, including Exhibit 99.1, is being furnished and shall not be deemed to '
        'be "filed" for purposes of Section 18 of the Securities Exchange Act of 1934, nor shall '
        'it be deemed incorporated by reference in any filing. Revenue grew 12% year over year.'
    )
    out = text.clean_text(raw, names)
    assert 'issued a press release announcing its financial results' in out
    assert 'furnished' not in out
    assert 'incorporated by reference' not in out
    assert 'Revenue grew 12%' in out


def test_clean_text_strips_forward_looking_statement_sentence():
    names = {'company_name': None, 'content_company_name': None, 'ticker': None}
    raw = (
        'Item 7.01 Regulation FD Disclosure. On April 1, 2025, the Company announced a new '
        'partnership expected to expand its market reach. This press release contains '
        'forward-looking statements within the meaning of the Private Securities Litigation '
        'Reform Act of 1995; investors should not place undue reliance on these statements. The '
        'partnership is expected to close in the second quarter.'
    )
    out = text.clean_text(raw, names)
    assert 'announced a new partnership' in out
    assert 'forward-looking statements' not in out
    assert 'undue reliance' not in out
    assert 'expected to close in the second quarter' in out


def test_clean_text_keeps_substantive_sentence_with_trailing_incorporated_by_reference_clause():
    # regression: a whole-sentence "incorporated by reference" filter was found (on a real-filing
    # smoke sample, before the DGX run) to discard the entire event sentence whenever a real 8-K
    # tacked "..., which is hereby incorporated by reference." onto its end -- only that trailing
    # clause is boilerplate, not the press-release content before it.
    names = {'company_name': None, 'content_company_name': None, 'ticker': None}
    raw = (
        'Item 2.02 Results of Operations and Financial Condition. On January 12, 2015, the '
        'Company issued a press release relating to its preliminary financial results for the '
        'fourth quarter of 2014, attached as Exhibit 99.1 to this Form 8-K, which is hereby '
        'incorporated by reference. This 8-K and Exhibit 99.1 shall not be deemed "filed" for '
        'purposes of Section 18 of the Securities Exchange Act of 1934, as amended.'
    )
    out = text.clean_text(raw, names)
    assert 'issued a press release relating to its preliminary financial results' in out
    assert 'attached as Exhibit 99.1 to this Form 8-K.' in out  # clause stripped, sentence still ends cleanly
    assert 'incorporated by reference' not in out
    assert 'shall not be deemed' not in out  # the second, purely-boilerplate sentence is still dropped


def test_clean_text_strips_not_deemed_filed_sentence_without_the_word_furnished():
    # the safe-harbor disclaimer commonly omits "furnished" entirely ("shall not be deemed
    # 'filed' for purposes of Section 18...") -- requiring "furnished" AND "not be deemed" missed
    # this common phrasing.
    names = {'company_name': None, 'content_company_name': None, 'ticker': None}
    raw = (
        'Item 7.01 Regulation FD Disclosure. On June 1, 2025, the Company posted an investor '
        'presentation to its website. This Current Report and the presentation shall not be '
        'deemed "filed" for purposes of Section 18 of the Securities Exchange Act of 1934.'
    )
    out = text.clean_text(raw, names)
    assert 'posted an investor presentation' in out
    assert 'shall not be deemed' not in out


def test_clean_text_cuts_tail_from_item_901_and_signature_block():
    names = {'company_name': None, 'content_company_name': None, 'ticker': None}
    raw = (
        'Item 5.02 Departure of Directors. On May 5, 2025, Jane Doe resigned as CFO effective '
        'immediately. Item 9.01 Financial Statements and Exhibits. (d) Exhibits. 99.1 Press '
        'release dated May 5, 2025. SIGNATURE Pursuant to the requirements of the Securities '
        'Exchange Act of 1934, the registrant has duly caused this report to be signed.'
    )
    out = text.clean_text(raw, names)
    assert 'Jane Doe resigned as CFO' in out
    assert 'Exhibits' not in out
    assert 'Pursuant to the requirements' not in out


# ---------------------------------------------------------------- score_texts

def _fake_score_texts(tok, model, texts, max_length=None, batch_size=None, device='cpu'):
    return np.tile(np.array([0.6, 0.1, 0.3], dtype=np.float32), (len(texts), 1))


def _fake_tok(batch, padding=True, truncation=True, max_length=None, return_tensors='pt'):
    import torch
    return {'input_ids': torch.zeros((len(batch), 4), dtype=torch.long)}


class _FakeOutput:
    def __init__(self, n):
        import torch
        self.logits = torch.tensor([[1.0, 0.0, 0.5]] * n)


def _fake_model(**enc):
    return _FakeOutput(enc['input_ids'].shape[0])


def test_score_texts_scores_empty_text_as_nan():
    # real score_texts (not mocked) exercises its own empty-text NaN handling, using a fake
    # tok/model (real torch tensors, no FinBERT weights) so no network/model load is needed.
    out = text.score_texts(_fake_tok, _fake_model, ['hello world', '', 'another doc'])
    assert np.isnan(out[1]).all()
    assert not np.isnan(out[0]).any()
    np.testing.assert_allclose(out[0], out[2])  # fake model returns the same logits regardless


# ---------------------------------------------------------------- fixtures for feature building

def _write_filings(path, rows):
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


@pytest.fixture
def synth_filings(tmp_path):
    rows = [
        dict(document_id='d1', permno=1, filing_date=pd.Timestamp('2020-01-15'),
             items=['1.01', '9.01'], company_name='A', content_company_name='A', ticker='AA'),
        dict(document_id='d2', permno=1, filing_date=pd.Timestamp('2020-01-20'),
             items=['1.01'], company_name='A', content_company_name='A', ticker='AA'),
        dict(document_id='d3', permno=1, filing_date=pd.Timestamp('2020-02-01'),
             items=['2.02'], company_name='A', content_company_name='A', ticker='AA'),
        dict(document_id='d4', permno=2, filing_date=pd.Timestamp('2020-01-31'),
             items=['8.01'], company_name='B', content_company_name='B', ticker='BB'),
    ]
    path = tmp_path / 'filings.parquet'
    _write_filings(path, rows)
    return path


def test_build_text_features_month_assignment_and_counts(monkeypatch, tmp_path, synth_filings):
    monkeypatch.setattr(config, 'FILINGS_PATH', synth_filings)
    monkeypatch.setattr(config, 'CACHE_DIR', tmp_path / 'cache')
    (tmp_path / 'cache').mkdir()

    feat = text.build_text_features()
    jan_permno1 = feat[(feat.permno == 1) & (feat.eom == pd.Timestamp('2020-01-31'))]
    assert len(jan_permno1) == 1
    assert jan_permno1['n_filings'].iloc[0] == 2  # d1, d2 only; d3 (Feb 1) must NOT count in January
    assert jan_permno1['item_1_01'].iloc[0] == 2

    feb_permno1 = feat[(feat.permno == 1) & (feat.eom == pd.Timestamp('2020-02-29'))]
    assert feb_permno1['n_filings'].iloc[0] == 1

    # tone columns omitted when the finbert scores file is absent, and nothing is cached to disk
    # (build_text_features is rebuilt on every call, never written to text_features.parquet)
    assert 'tone_mean' not in feat.columns
    assert not (tmp_path / 'cache' / 'text_features.parquet').exists()


def test_build_text_features_includes_tone_when_scores_present(monkeypatch, tmp_path, synth_filings):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'FILINGS_PATH', synth_filings)
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)

    scores = pd.DataFrame({
        'document_id': ['d1', 'd2', 'd3', 'd4'],  # covers every document_id in synth_filings
        'permno': [1, 1, 1, 2],
        'filing_date': [pd.Timestamp('2020-01-15'), pd.Timestamp('2020-01-20'),
                         pd.Timestamp('2020-02-01'), pd.Timestamp('2020-01-31')],
        'fb_pos': [0.8, 0.2, 0.5, 0.1],
        'fb_neg': [0.1, 0.7, 0.3, 0.8],
        'fb_neu': [0.1, 0.1, 0.2, 0.1],
        'max_length': text.MAX_LENGTH,
        'device': 'cpu',
        'dtype': 'fp32',
        'revision': text.FINBERT_REVISION,
    })
    scores.to_parquet(text.scores_path_for(text.MAX_LENGTH), index=False)

    feat = text.build_text_features()
    assert 'tone_mean' in feat.columns
    jan_permno1 = feat[(feat.permno == 1) & (feat.eom == pd.Timestamp('2020-01-31'))].iloc[0]
    expected_tone_mean = np.mean([0.8 - 0.1, 0.2 - 0.7])
    assert jan_permno1['tone_mean'] == pytest.approx(expected_tone_mean)
    assert not (cache_dir / 'text_features.parquet').exists()  # no on-disk cache


@pytest.mark.parametrize('bad_col,bad_value', [('max_length', 128), ('revision', 'some-other-commit')])
def test_build_text_features_raises_when_scores_settings_mismatch(monkeypatch, tmp_path, synth_filings, bad_col, bad_value):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'FILINGS_PATH', synth_filings)
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)

    scores = pd.DataFrame({
        'document_id': ['d1', 'd2', 'd3', 'd4'],
        'permno': [1, 1, 1, 2],
        'filing_date': [pd.Timestamp('2020-01-15'), pd.Timestamp('2020-01-20'),
                         pd.Timestamp('2020-02-01'), pd.Timestamp('2020-01-31')],
        'fb_pos': [0.8, 0.2, 0.5, 0.1],
        'fb_neg': [0.1, 0.7, 0.3, 0.8],
        'fb_neu': [0.1, 0.1, 0.2, 0.1],
        'max_length': text.MAX_LENGTH,
        'device': 'cpu',
        'dtype': 'fp32',
        'revision': text.FINBERT_REVISION,
    })
    scores[bad_col] = bad_value
    scores.to_parquet(text.scores_path_for(text.MAX_LENGTH), index=False)

    with pytest.raises(AssertionError):
        text.build_text_features()


def test_build_text_features_raises_when_scores_missing_document_ids(monkeypatch, tmp_path, synth_filings):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'FILINGS_PATH', synth_filings)
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)

    # missing d3, d4 -- an incomplete scores file must not be silently used
    scores = pd.DataFrame({
        'document_id': ['d1', 'd2'],
        'permno': [1, 1],
        'filing_date': [pd.Timestamp('2020-01-15'), pd.Timestamp('2020-01-20')],
        'fb_pos': [0.8, 0.2], 'fb_neg': [0.1, 0.7], 'fb_neu': [0.1, 0.1],
    })
    scores.to_parquet(text.scores_path_for(text.MAX_LENGTH), index=False)

    with pytest.raises(AssertionError):
        text.build_text_features()


def test_truncation_invariance(monkeypatch, tmp_path, synth_filings):
    monkeypatch.setattr(config, 'FILINGS_PATH', synth_filings)
    monkeypatch.setattr(config, 'CACHE_DIR', tmp_path / 'cache_full')
    (tmp_path / 'cache_full').mkdir()
    full_feat = text.build_text_features()

    # truncated dataset: drop the February filing (d3), keep everything <= end of January
    truncated_rows = pd.read_parquet(synth_filings)
    truncated_rows = truncated_rows[truncated_rows['filing_date'] <= pd.Timestamp('2020-01-31')]
    trunc_path = tmp_path / 'filings_trunc.parquet'
    truncated_rows.to_parquet(trunc_path, index=False)
    monkeypatch.setattr(config, 'FILINGS_PATH', trunc_path)
    monkeypatch.setattr(config, 'CACHE_DIR', tmp_path / 'cache_trunc')
    (tmp_path / 'cache_trunc').mkdir()
    trunc_feat = text.build_text_features()

    full_jan = full_feat[full_feat.eom == pd.Timestamp('2020-01-31')].sort_values('permno').reset_index(drop=True)
    trunc_jan = trunc_feat[trunc_feat.eom == pd.Timestamp('2020-01-31')].sort_values('permno').reset_index(drop=True)
    pd.testing.assert_frame_equal(full_jan, trunc_jan)


def test_add_text_features_zero_fill_and_has_filing(monkeypatch, tmp_path, synth_filings):
    monkeypatch.setattr(config, 'FILINGS_PATH', synth_filings)
    monkeypatch.setattr(config, 'CACHE_DIR', tmp_path / 'cache')
    (tmp_path / 'cache').mkdir()

    panel = pd.DataFrame({
        'permno': [1, 1, 2, 3],
        'eom': [pd.Timestamp('2020-01-31'), pd.Timestamp('2020-02-29'),
                pd.Timestamp('2020-01-31'), pd.Timestamp('2020-01-31')],
    })
    out = text.add_text_features(panel)

    assert 'has_filing' not in text.TEXT_FEATURES  # excluded per survivorship-leak note
    assert 'has_filing' in out.columns

    row3 = out[(out.permno == 3) & (out.eom == pd.Timestamp('2020-01-31'))].iloc[0]
    assert row3['has_filing'] == 0
    assert row3['n_filings'] == 0
    for c in text.TEXT_FEATURES:
        if c in out.columns:  # tone_* absent here since no finbert scores file in this test
            assert row3[c] == 0

    row1_jan = out[(out.permno == 1) & (out.eom == pd.Timestamp('2020-01-31'))].iloc[0]
    assert row1_jan['has_filing'] == 1
    assert row1_jan['n_filings'] == 2


# ---------------------------------------------------------------- filing_coverage_by_exit

def test_filing_coverage_by_exit_buckets_and_censoring(monkeypatch, tmp_path):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    monkeypatch.setattr(config, 'CACHE_DIR', cache_dir)

    # one filing, matching permno 1's row 2 months before its exit ("1-2 months before exit").
    filings_rows = [dict(document_id='f1', permno=1, filing_date=pd.Timestamp('2020-08-15'), items=['8.01'])]
    filings_path = tmp_path / 'filings.parquet'
    pd.DataFrame(filings_rows).to_parquet(filings_path, index=False)
    monkeypatch.setattr(config, 'FILINGS_PATH', filings_path)

    # panel_end (from the RAW characteristics data) = 2021-12-31.
    # permno 1: last RAW eom 2020-10-31, 14 months before panel_end -> confirmed exit; its universe
    #   rows sit at 13/11/6/5/2/0 months before that exit, covering every bucket.
    # permno 2: last RAW eom IS panel_end -> confirmed survivor (both of its universe rows count).
    # permno 3: last RAW eom 2021-07-31, 5 months before panel_end -> ambiguous, fully censored.
    # permno 4: its only UNIVERSE row is way back in 2019 (it later drops out of the universe on
    #   size/price), but it keeps appearing in the RAW data through panel_end -- a temporary
    #   universe drop-out, not a real exit, so its universe row must land in "survivor", not any
    #   exit bucket (this is the reason exit/survivor status comes from the raw data, not the
    #   universe-filtered panel).
    panel = pd.DataFrame({
        'permno': [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4],
        'eom': [
            pd.Timestamp('2019-09-30'),  # 13 months before exit -> 12+
            pd.Timestamp('2019-11-30'),  # 11 months before exit -> 6-11
            pd.Timestamp('2020-04-30'),  # 6 months before exit -> 6-11
            pd.Timestamp('2020-05-31'),  # 5 months before exit -> 3-5
            pd.Timestamp('2020-08-31'),  # 2 months before exit -> 1-2 (has the filing)
            pd.Timestamp('2020-10-31'),  # last row -> last panel row
            pd.Timestamp('2021-06-30'),  # survivor, mid-history
            pd.Timestamp('2021-12-31'),  # survivor, last row == panel_end
            pd.Timestamp('2021-01-31'),  # permno 3: censored
            pd.Timestamp('2021-07-31'),  # permno 3: censored (last row, ambiguous)
            pd.Timestamp('2019-01-31'),  # permno 4: temporary universe drop-out, not a real exit
        ],
    })
    monkeypatch.setattr('src.data.build_panel', lambda: panel)

    raw_chars = pd.DataFrame({
        'permno': [1, 2, 3, 4],
        'eom': [pd.Timestamp('2020-10-31'), pd.Timestamp('2021-12-31'),
                pd.Timestamp('2021-07-31'), pd.Timestamp('2021-12-31')],
    })
    monkeypatch.setattr('src.data.load_chars', lambda columns=None: raw_chars)

    out = text.filing_coverage_by_exit()

    assert list(out['bucket']) == [
        'last panel row', '1-2 months before exit', '3-5 months before exit',
        '6-11 months before exit', '12+ months before exit', 'survivor to panel end',
    ]
    n_by_bucket = dict(zip(out['bucket'], out['n']))
    assert n_by_bucket == {
        'last panel row': 1, '1-2 months before exit': 1, '3-5 months before exit': 1,
        '6-11 months before exit': 2, '12+ months before exit': 1, 'survivor to panel end': 3,
    }
    assert sum(n_by_bucket.values()) == 9  # 11 universe rows, 2 censored (permno 3)

    cov_by_bucket = dict(zip(out['bucket'], out['coverage']))
    assert cov_by_bucket['1-2 months before exit'] == pytest.approx(1.0)  # the one filing lands here
    for bucket in ['last panel row', '3-5 months before exit', '6-11 months before exit',
                   '12+ months before exit', 'survivor to panel end']:
        assert cov_by_bucket[bucket] == pytest.approx(0.0)


# ---------------------------------------------------------------- score_finbert (mocked FinBERT)

@pytest.fixture
def synth_filings_for_score(tmp_path):
    rows = []
    for i in range(7):
        rows.append(dict(
            document_id=f'doc{i}', permno=100 + (i % 2),
            filing_date=pd.Timestamp('2021-01-01') + pd.Timedelta(days=i),
            items=['8.01'], text=f'Item 8.01 filler text number {i}.',
            company_name='Foo Inc', content_company_name='Foo Inc', ticker='FOO',
        ))
    path = tmp_path / 'filings_score.parquet'
    _write_filings(path, rows)
    return path


def test_score_finbert_chunks_resume_and_consolidate(monkeypatch, tmp_path, synth_filings_for_score):
    monkeypatch.setattr(config, 'FILINGS_PATH', synth_filings_for_score)
    monkeypatch.setattr(config, 'CACHE_DIR', tmp_path / 'cache')
    (tmp_path / 'cache').mkdir()
    monkeypatch.setattr(text, 'CHUNK_SIZE', 3)

    calls = {'n': 0}

    def load_calls(*a, **k):
        calls['n'] += 1
        return 'TOK', 'MODEL'

    monkeypatch.setattr(text, '_load_finbert', load_calls)
    monkeypatch.setattr(text, 'score_texts', _fake_score_texts)

    text.score_finbert(max_length=128, device='cpu')

    chunk_dir = tmp_path / 'cache' / 'finbert_chunks_L128'
    parts = sorted(chunk_dir.glob('part_*.parquet'))
    assert len(parts) == 3  # ceil(7/3)
    consolidated = tmp_path / 'cache' / 'finbert_scores_L128.parquet'
    assert consolidated.exists()
    out = pd.read_parquet(consolidated)
    assert len(out) == 7
    assert set(out.columns) >= {'document_id', 'permno', 'filing_date', 'fb_pos', 'fb_neg', 'fb_neu',
                                 'max_length', 'device', 'dtype', 'revision'}
    assert out['document_id'].is_unique
    assert (out['max_length'] == 128).all()
    assert (out['device'] == 'cpu').all()
    assert (out['revision'] == text.FINBERT_REVISION).all()

    # resumability: delete one chunk, re-run, only that chunk gets rebuilt (model loaded once more)
    calls['n'] = 0
    consolidated.unlink()
    parts[1].unlink()
    text.score_finbert(max_length=128, device='cpu')
    assert calls['n'] == 1
    parts_after = sorted(chunk_dir.glob('part_*.parquet'))
    assert len(parts_after) == 3
    assert consolidated.exists()
    assert len(pd.read_parquet(consolidated)) == 7


def test_chunk_dir_for_separates_max_lengths(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'CACHE_DIR', tmp_path / 'cache')
    (tmp_path / 'cache').mkdir()

    assert text.chunk_dir_for(128) == tmp_path / 'cache' / 'finbert_chunks_L128'
    assert text.chunk_dir_for(512) == tmp_path / 'cache' / 'finbert_chunks_L512'


def test_score_finbert_device_max_length_plumbing_cpu(monkeypatch, tmp_path, synth_filings_for_score):
    """Device/max_length are new flags; on CPU with a mocked model they must not change the
    scores that come out, only which chunk_dir/output file/columns record the setting used."""
    monkeypatch.setattr(config, 'FILINGS_PATH', synth_filings_for_score)
    monkeypatch.setattr(config, 'CACHE_DIR', tmp_path / 'cache')
    (tmp_path / 'cache').mkdir()
    monkeypatch.setattr(text, 'CHUNK_SIZE', 10)
    monkeypatch.setattr(text, '_load_finbert', lambda *a, **k: ('TOK', 'MODEL'))
    monkeypatch.setattr(text, 'score_texts', _fake_score_texts)

    text.score_finbert(max_length=128, device='cpu', batch_size=4)
    out_128 = pd.read_parquet(tmp_path / 'cache' / 'finbert_scores_L128.parquet').sort_values('document_id')

    text.score_finbert(max_length=256, device='cpu', batch_size=16)
    out_256 = pd.read_parquet(tmp_path / 'cache' / 'finbert_scores_L256.parquet').sort_values('document_id')

    shared_cols = ['document_id', 'permno', 'filing_date', 'fb_pos', 'fb_neg', 'fb_neu']
    pd.testing.assert_frame_equal(
        out_128[shared_cols].reset_index(drop=True), out_256[shared_cols].reset_index(drop=True)
    )
    assert (out_128['max_length'] == 128).all()
    assert (out_256['max_length'] == 256).all()


# ---------------------------------------------------------------- slow, real-data tests

@pytest.mark.slow
def test_build_text_features_real_data_shape_and_bounds():
    feat = text.build_text_features()
    dup = feat.duplicated(['permno', 'eom'])
    assert not dup.any()
    assert feat['eom'].min() >= pd.Timestamp('2015-01-31')
    assert feat['eom'].max() <= pd.Timestamp('2026-08-31')


@pytest.mark.slow
def test_filing_coverage_by_exit_runs():
    out = text.filing_coverage_by_exit()
    assert list(out.columns) == ['bucket', 'n', 'coverage']
    assert len(out) == 6
    assert (out['coverage'].dropna() >= 0.0).all() and (out['coverage'].dropna() <= 1.0).all()
    assert (out['n'] > 0).all()
