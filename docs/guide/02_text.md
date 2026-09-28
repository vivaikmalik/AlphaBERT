# Chapter 2 — Text: 8-K Filings and FinBERT Sentiment

This chapter covers `src/text.py`: how the raw 8-K filing archive becomes a handful of
per-stock-month numbers the model can use. Four things happen, in order: (1) each filing's raw
text is cleaned down to an "event body"; (2) FinBERT scores that body for tone; (3) scores are
aggregated to `(permno, eom)`; (4) a documented survivorship trap in the raw archive is defused so
those numbers don't leak future information about which firms are still alive.

Every code citation below is `file:function` or `file:line`. Every number is either read directly
from a source file (labelled with its origin) or measured by me against the real filing archive
during the writing of this chapter (labelled **measured**). Nothing here is estimated or invented.

## 1. The 8-K dataset

### What an 8-K is

SEC Form 8-K is a "current report": whenever a public company has certain reportable events, it
must file (or, for some items, "furnish") an 8-K disclosing them, generally within four business
days of the event, "subject to item-specific exceptions" (brief, p.10). Unlike 10-K/10-Q, which
report on a fixed calendar schedule, 8-Ks are event-driven — a company might file several in one
week around an earnings call and acquisition, then nothing for months. The brief's framing (p.10)
is explicit that this is a change from prior years' periodic-filing datasets to an *event-filing*
dataset, which is why the whole module is organized around dated events rather than fixed-frequency
reports.

Each 8-K carries a list of one or more **item codes** identifying which reportable event(s) it
covers (readme.md section 3: "A single filing may cover several items"). `config.KEY_ITEMS`
(`src/config.py:34`) is the fixed list of item codes the pipeline tracks as individual features:

```python
KEY_ITEMS = ['1.01', '1.02', '2.01', '2.02', '2.05', '2.06', '3.01', '4.01', '4.02', '5.02', '7.01', '8.01']
```

What each means (descriptions for every code except 3.01 are the brief's own, p.10; 3.01's is the
standard SEC item description, since the brief doesn't cover it):

| Item | Meaning |
|---|---|
| 1.01 | Entry into a material definitive agreement — contracts signed, credit facilities, partnerships, supply deals |
| 1.02 | Termination of a material definitive agreement |
| 2.01 | Completion of acquisition or disposition of assets |
| 2.02 | Results of operations and financial condition — often the earnings announcement itself (the press release may be an exhibit, not part of the supplied main text) |
| 2.05 | Costs associated with exit or disposal activities (restructuring charges) |
| 2.06 | Material impairments (write-downs) |
| 3.01 | Notice of delisting or failure to satisfy a continued listing rule; transfer of listing |
| 4.01 | Changes in the registrant's certifying accountant (auditor changes) |
| 4.02 | Non-reliance on previously issued financial statements or a related audit report — "two of the strongest distress signals in the entire corpus" per the brief |
| 5.02 | Officer/director changes — CEO, CFO, board turnover, "including the abrupt kind" |
| 7.01 | Regulation FD disclosure — guidance updates, buybacks, litigation, product announcements |
| 8.01 | Other events — catch-all for anything else management discloses |

`build_text_features()` (`src/text.py:449`) turns each code into a per-stock-month count column
named `item_<code with '.' replaced by '_'>`, e.g. `2.02` → `item_2_02`.

### `filing_date` as the availability clock

The filing archive carries several date-like fields, and the readme is emphatic that most of them
are *not* trustworthy availability timestamps:
- `filing_date`: "Preserved provider filing date. **Not independently verified** against SEC
  acceptance records" (readme.md section 3).
- `filing_timestamp_utc` / `filing_time_precision`: every filing in this release has
  `filing_time_precision = 'date_or_midnight_placeholder'` (readme.md section 1) — there is no real
  intraday timestamp.
- `sec_accepted_at_utc`: "Entirely null in this snapshot" (readme.md section 3) — the one field that
  would give a verified SEC acceptance time simply isn't populated.
- `content_report_date`: "Date of report/earliest event extracted from the filing header. **Not a
  public availability date**" (readme.md section 3) — this is when the underlying *event* happened,
  which can precede the filing by days or weeks; it says nothing about when the market could have
  read about it.

Given that, `filing_date` — imprecise (date-level, unverified) as it is — is the least-bad clock
available, and the brief is direct about the obligation this creates (p.11): *"Filing timing must
respect the portfolio's information cutoff. A disclosure dated the 3rd cannot be used to form a
portfolio on the 1st... State a conservative availability convention."* `text.py` states its
convention as a code comment and enforces it structurally: `eom = filing_date + MonthEnd(0)`
(`src/text.py:460`), i.e. every filing is attributed to the calendar month of its `filing_date`, and
SPEC section 4's invariant is "features at eom t use only filings with `filing_date` in calendar
month t" (`docs/SPEC.md:48`).

### Month-t aggregation and its conservatism

The brief licenses exactly this design (p.11): *"For the current monthly frequencies, you can
accumulate the news and extract their sentiment within the current month t to make any forecasts
for the month t+1."* Concretely, `build_text_features()` groups all filings whose `filing_date`
falls in calendar month t into the `eom = t` row (`src/text.py:456-462`), and `add_text_features`
left-joins that onto the panel on `(permno, eom)` (`src/text.py:513-529`). Per SPEC section 2,
positions are formed at the *end* of month t and held during month t+1 (`docs/SPEC.md:26`), so a
filing dated any day in month t — including the last calendar day — is still dated no later than
the formation date. No filing dated in month t+1 ever reaches the eom=t feature row.

This is conservative in a specific, narrow sense: it never uses information dated after the
formation date, but it is *not* conservative about intraday timing — a filing whose date-only
timestamp happens to be the formation day itself is treated as fully available at formation, even
though `filing_time_precision` can't actually distinguish "filed at 8am" from "filed at 11:58pm."
The design accepts that residual imprecision (the brief permits it: "state a conservative
availability convention," not "prove intraday availability") rather than discarding the last few
days of every month, which would be a different and much more aggressive form of conservatism the
brief doesn't require. `docs/SPEC.md:48` states the invariant precisely, and
`tests/test_text.py::test_truncation_invariance` / `tests/test_integrity.py::test_truncation_invariance_text_real`
verify it holds mechanically (section 6 below).

One more caveat inherited from readme.md section 3: "This does not imply a filing exists for every
stock-month." Most `(permno, eom)` rows in the panel have zero 8-Ks in that month. How the pipeline
handles that absence — rather than treating it as missing data — is the subject of section 5.

## 2. `clean_text` step by step

`clean_text(text, names)` (`src/text.py:168-220`) turns a raw filing (cover page, legal
boilerplate, exhibit list, signature block and all) into the shortest text that still contains the
actual disclosed event, so FinBERT's limited token budget (section 3) is spent on substance. The
function docstring states the goal directly: *"Returns '' only when the filing genuinely has no
usable text left."* Below is every step, in the order the code actually executes them.

### Step 1 — locate the body, drop the cover page

`_ITEM_RE = re.compile(r'I\s*t\s*e\s*m\.?\s{0,3}\d\s{0,2}\.\s{0,2}\d\s{0,2}\d', re.IGNORECASE)`
(`src/text.py:52`) matches an item header like "Item 5.02" while tolerating odd whitespace inside
the digits (real filings occasionally have "Item 5.0 7" — the regex is built to still match that;
`tests/test_text.py::test_clean_text_tolerates_malformed_item_header_spacing` covers it).

- If found, the body starts at that match — everything before it (registrant name, CIK, "Check the
  appropriate box," "Emerging growth company [ ]," etc.) is dropped.
- If not found and the text is longer than `_COVER_FALLBACK_CHARS = 1500` chars, the first 1,500
  characters are dropped unconditionally as a presumed cover page.
- If not found and the text is 1,500 chars or shorter, it's too short to safely assume the opening
  is a cover page (dropping it would empty the text), so the whole thing is kept as-is.

### Step 2 — table-of-contents guard

Some 8-Ks print a navigation block ("TABLE OF CONTENTS Item 5.02 ... 3 Item 9.01 ... SIGNATURE 4
...") before the real section header repeats the same "Item X.XX ..." text. Found on a real-filing
sample before the DGX run, in about 1.7% of filings (`src/text.py:56`). If `_TOC_RE` (`"TABLE OF
CONTENTS"`, case-insensitive) matches anywhere before the first `_ITEM_RE` hit, `clean_text` looks
for the *exact same matched text* recurring later in the document and, if found, restarts the body
there instead — otherwise the "body" would just be the TOC/SIGNATURES/EXHIBIT INDEX navigation
text, which is strictly worse than losing the item title.

I confirmed this on a real filing while writing this chapter (`document_id
MF-CR-20151013-NOC-UN`, Northrop Grumman, filed 2015-10-13). The raw text contains:

```
... TABLE OF CONTENTS Item 5.02 Departure of Directors or Certain Officers; ... 3
Item 9.01 Financial Statements and Exhibits 3 SIGNATURE 4 INDEX TO EXHIBITS 5 EXHIBIT 99.1 - 2 -
Item 5.02 Departure of Directors or Certain Officers; ... (c)(e) On October 14, 2015,
Northrop Grumman Corporation (the "Company") ...
```

`clean_text` correctly skips past the TOC entry and starts the body at the second "Item 5.02"
occurrence, so the cleaned text leads with "Item 5.02 ... (c)(e) On October 14, 2015, the Company
(the "Company") announced ..." rather than the page-number-riddled navigation block.

### Step 3 — cut the tail

`_TAIL_CUT_RE` (`src/text.py:73-78`) matches the first occurrence of any of: "Item 9.01" (any
spacing variant, same tolerance as `_ITEM_RE`), the word "SIGNATURE(S)", or "Pursuant to the
requirements of the Securities Exchange Act." Everything from that point to the end of `body` is
discarded. Item 9.01 is the "Financial Statements and Exhibits" item — an exhibit list, never
itself a signal — and everything after it is boilerplate signature-block text. This cut happens
*before* the title-skip and boilerplate-sentence steps below, which matters (see the MIR example).

### Step 4 — skip the boilerplate item title

Most item bodies read "Item X.XX <boilerplate title>. On <date>, the Company...". Skipping the
title packs more of the actual event into FinBERT's fixed token budget. `_TITLE_SKIP_RE = re.compile(r'\.\s+(?=On\s)')`
looks, only within the first `_TITLE_SKIP_WINDOW = 300` characters of the body, for a period
followed by whitespace and then "On" — the start of the "On <date>, ..." event sentence. If found,
everything up to and including that period+whitespace is dropped. **If not found, the title is kept
— a deliberate safe fallback rather than risking a cut into real content on an unusual filing**
(`src/text.py:63-66`).

I found two real, unforced examples of this fallback while testing:
- **DENTSPLY SIRONA** (`MF-CR-20251002-XRAY-UW`, item 5.02): the title runs "...Compensatory
  Arrangements of Certain Officers" directly into "On October 2, 2025, the Company..." with **no
  period** before "On" (the title itself has no terminating punctuation), so `_TITLE_SKIP_RE`
  doesn't match and the full title survives in the cleaned output.
- **American Water Works** (`MF-CR-20210225-AWK-UN`, item 8.01): the title is a long descriptive
  headline ("...Approval of Settlement Agreement in Pennsylvania-American Water Company General
  Rate Case") that also runs straight into "On February 25, 2021" with no preceding period — same
  fallback, title kept.

Both confirm the fallback triggers on real data, not just the synthetic case in
`tests/test_text.py::test_clean_text_keeps_title_when_no_on_sentence_nearby`.

### Step 5 — strip the "incorporated by reference" clause

Real 8-Ks routinely end a substantive sentence with a trailing legal clause: "...attached as
Exhibit 99.1 ..., which is hereby incorporated by reference." An earlier version of the cleaner
filtered this as a *whole-sentence* boilerplate pattern, which was found — scoring a real-filing
sample before the DGX run — to discard the entire event sentence, not just the boilerplate clause
(`src/text.py:88-95`, `docs/research_log.md` 2026-09-27 entry). The fix,
`_INCORPORATED_BY_REFERENCE_CLAUSE_RE` (`src/text.py:106-110`), matches only the trailing clause
itself:

```
,?\s*(?:which is |and is |is\s+)?hereby\s+incorporated\s+(?:herein\s+)?by\s+reference(?:\s+in\s+its\s+entirety)?\.?
```

and replaces it with a single `.`, so the sentence's real content and its terminating period
survive. `tests/test_text.py::test_clean_text_keeps_substantive_sentence_with_trailing_incorporated_by_reference_clause`
verifies this precisely.

**A real-data limitation I found while testing, worth stating plainly:** this regex requires the
literal word "hereby." A real filing (`MF-CR-20240801-MIR-UN`, Mirion Technologies, item 2.02)
phrases the clause without it — "...is furnished herewith as Exhibit 99.1 **and incorporated
herein by reference**." (no "hereby") — and the clause is *not* stripped; it survives untouched in
the cleaned output. This isn't a contradiction of the design intent (the pattern is deliberately
narrow to avoid the whole-sentence-deletion bug above), but it means the coverage of this specific
clause-stripping step is narrower in practice than "incorporated-by-reference sentences" might
suggest — it catches the "hereby incorporated ... by reference" phrasing specifically, not every
incorporation-by-reference wording.

### Step 6 — strip boilerplate sentences

After splitting on sentence boundaries (`_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')`),
`_is_boilerplate_sentence` (`src/text.py:113-122`) drops any sentence matching either of two
patterns, measured on a 4% filing sample (`src/text.py:19-21, 83-84`; Item 2.02 windows — 33% of
the corpus — were ≥2-phrase safe-harbor/furnished boilerplate 82.5% of the time; 7.01, 55%; 8.01,
30%):

1. **Furnished / not-deemed-filed disclaimer**: `_NOT_DEEMED_RE` ("not be deemed") matched together
   with either `_FURNISHED_RE` ("furnished") or `_FILED_RE` ("filed", word-boundaried). Requiring
   "not be deemed" **and** ("furnished" **or** "filed") — rather than "furnished" alone — is
   deliberate: the common phrasing "shall not be deemed 'filed' for purposes of Section 18..."
   never uses the word "furnished" at all, so an earlier furnished-only rule missed it
   (`src/text.py:114-118`).
2. **Forward-looking-statements safe harbor**: any of "forward-looking statements?", "Private
   Securities Litigation Reform Act", or "undue reliance".

Because this filter runs on sentence-split text (period-delimited), it only removes whole
sentences that match — it will not partially truncate a sentence. Step 3 (tail cut), by contrast,
truncates at a raw character offset regardless of sentence boundaries, which produces a different
artifact worth noting: in the Mirion example above, the safe-harbor sentence itself — "The
information contained in this Item 2.02 **and Item 9.01** in this Current Report on Form 8-K...
shall not be deemed 'filed'..." — cites "Item 9.01" *by reference*, mid-sentence, before the actual
Item 9.01 section header appears later in the document. `_TAIL_CUT_RE` fires on that first, earlier
mention and truncates the body right there — mid-sentence — rather than the sentence-boilerplate
filter getting a chance to drop the (still-incomplete) sentence cleanly. The result is a dangling
fragment at the very end of the cleaned text: "...furnished herewith as Exhibit 99.1 and
incorporated herein by reference. The information contained in this Item 2.02 and". This still
satisfies the tail-cut's literal goal (nothing from the real Item 9.01 exhibit list or signature
block leaks in), but it's a real, observed interaction between the two steps' different units of
truncation (character offset vs. sentence), not something either step's own tests exercise on
synthetic data.

### Step 7 — name/ticker masking

`_name_variants(names)` (`src/text.py:150-165`) builds two sets from `company_name` and
`content_company_name`:
- **Full variants**: the name exactly as supplied, matched **case-insensitively**.
- **Short variants**: the name with common legal suffixes stripped (`_strip_suffix`: "Inc",
  "Corp", "Corporation", "Co", "Ltd", "LLC", "L.P.", trailing parenthetical/slash jurisdiction
  tags like `/MA/`, applied repeatedly until stable), matched **case-sensitively, exactly as
  written** — and only kept if the stripped name is "long/distinctive enough": at least 2 words,
  or at least 6 characters (`src/text.py:154, 163`).

The code comment is explicit about why the short-variant gate exists: unconditional whole-word
masking on a bare name/ticker collides with ordinary words — "Target, Box, Post, Team, MA..."
(`src/text.py:133-136`). `tests/test_text.py::test_clean_text_short_name_variant_needs_two_words_or_six_chars`
shows "Box, Inc." → short variant "Box" is rejected (3 chars, 1 word), so "Please box the
equipment" survives untouched; `test_clean_text_short_name_variant_masked_when_long_or_multiword`
shows "Target Corporation" → short variant "Target" (6 chars) *is* kept and masked. The
case-sensitivity matters too: `test_clean_text_short_variant_is_case_sensitive` shows "Acme
Widgets" (Title Case, masked) coexisting with "acme widgets brand mugs" (lowercase, untouched) in
the same document.

Tickers are handled separately and even more narrowly. `_TICKER_CTX_RE` only matches a ticker
immediately preceded by an explicit exchange marker — "NYSE", "NYSE American", "Nasdaq"/"NASDAQ",
or "ticker symbol" (`src/text.py:137`). A bare ticker mention with no such marker is left alone.
The motivating false positive, stated directly in the code: unconditional bare-ticker masking was
"also masking 'Form 8-K' when ticker == 'FORM'" (`src/text.py:136`). **I confirmed this is a live
case in the actual dataset, not a hypothetical**: 90 filings (measured) carry `ticker == 'FORM'`
(FormFactor Inc). Every one of those 90 filings' boilerplate opens with "...intended to
simultaneously satisfy the filing obligation..." and the literal phrase "Form 8-K" appears
repeatedly in the cover-page/legal boilerplate — bare-word ticker masking would have corrupted
"Form 8-K" into "the Company 8-K" throughout all of them (the cover page is usually dropped by step
1, but the phrase also appears in retained boilerplate, e.g. incorporation-by-reference clauses
referencing "this Form 8-K"). `tests/test_text.py::test_clean_text_ticker_masked_only_in_exchange_context`
covers the corresponding regression on a synthetic "MA" ticker (colliding with "Massachusetts").

Both replacements use `(?<!\w)...(?!\w)` boundaries rather than `\b` on both sides — the code
comment explains: a plain `\b` fails immediately after a variant ending in punctuation (e.g.
"Inc.") followed by a space, since both sides are already non-word characters (`src/text.py:205-206`).

**A real-data artifact from this exact boundary choice, which I found and verified**: for American
Water Works (`MF-CR-20210225-AWK-UN`), `company_name = 'American Water Works Company, Inc.'` —
*including* the trailing period. The full-variant regex is built from this literal string via
`re.escape(variant)`, so the match consumes the trailing "." as part of the company name itself.
Replacing the whole match (name **and** its period) with "the Company" (no period) silently
deletes what was functioning as the sentence's terminating period. Raw: "...a subsidiary of
American Water Works Company, Inc. Under the PaPUC's order, ..."; cleaned: "...a subsidiary of the
Company Under the PaPUC's order, ..." — two sentences run together with no punctuation between
them. This is a minor, narrow-scope artifact (it only occurs when the supplied `company_name`
field itself ends in a period that doubles as a sentence-ending period in the source text), not a
correctness bug in the sense of masking the wrong content, but it does show up as a genuine
readability wrinkle in real cleaned output.

### Step 8 — collapse whitespace, empty → `''`

The final line, `re.sub(r'\s+', ' ', body).strip()`, collapses all whitespace (including line
breaks from tables, address blocks, or paragraph breaks) to single spaces — the readme's own
caveat that "whitespace normalization can flatten table layout" (readme.md section 3) applies here
too. If nothing survives all the above steps, `clean_text` returns `''`. Critically, an empty
string is **not** scored as neutral text — `score_texts` (section 3) maps it to `NaN`, which
pandas aggregation later excludes rather than silently treating as tone-neutral
(`src/text.py:264-266`).

### A short before/after example

Real filing `MF-CR-20240801-MIR-UN` (Mirion Technologies, item 2.02, filed 2024-08-01). Raw text
opens with ~140 words of SEC cover-page boilerplate (agency header, registrant address, checkbox
disclosures, "Emerging growth company ☐", exchange listing table) before the first item header.
Cleaned output, in full:

> *"On August 1, 2024, the Company (the "Company") issued a press release announcing its financial
> results for the fiscal quarter ended June 30, 2024. A copy of the press release is furnished
> herewith as Exhibit 99.1 and incorporated herein by reference. The information contained in this
> Item 2.02 and"*

Raw length 3,387 characters → cleaned length 298 characters (measured). The cover page, item
title ("Item 2.02 Results of Operations and Financial Condition."), Item 9.01 exhibit list, and
signature block are all gone; the actual disclosed event ("issued a press release announcing its
financial results...") leads the text, exactly as the module docstring intends
(`src/text.py:3-4`: "so FinBERT reads the actual event description first, not a press-release
exhibit"). The trailing dangling fragment ("...Item 2.02 and") is the tail-cut/sentence-boundary
interaction described in Step 6 above; the doubled "the Company (the "Company")" is the ordinary
consequence of masking a name that the filing itself immediately re-introduces via its own "(the
'Company')" defined-term convention — both are genuine, non-fatal artifacts of a deliberately
narrow, false-positive-averse design, not signs the cleaning failed.

## 3. FinBERT

### What it is, and its label order

`FINBERT_MODEL = 'ProsusAI/finbert'` (`src/text.py:41`) is a BERT-base model fine-tuned for
financial-text sentiment. Per its own model card / config (I read
`outputs/hf_cache/hub/models--ProsusAI--finbert/snapshots/<pinned revision>/config.json`, a local
cached copy of the pinned snapshot, to verify this rather than trust memory) and per the module
docstring (`src/text.py:12-14`), it was built in three stages: BERT-base pretrained on general
2018-era corpora (BooksCorpus + Wikipedia), further pretrained on Reuters TRC2 (2008–2010
newswire), then fine-tuned for sentiment classification on the Financial PhraseBank (labelled
2014).

The cached `config.json` confirms the label order the code relies on:

```json
"id2label": {"0": "positive", "1": "negative", "2": "neutral"},
"label2id": {"positive": 0, "negative": 1, "neutral": 2}
```

`score_texts` (`src/text.py:261-290`) returns an `(n, 3)` array and documents it as "[pos, neg,
neu] probabilities, in id2label order 0/1/2" — matching `config.json` exactly, so
`out[:, 0]`/`fb_pos`, `out[:, 1]`/`fb_neg`, `out[:, 2]`/`fb_neu` are labelled correctly downstream
(`src/text.py:371-373`).

### From logits to probabilities

The model outputs three logits $z_0, z_1, z_2$ (one per label); `score_texts` converts them to
probabilities with softmax,
$$P(\text{label}=k \mid x) = \frac{e^{z_k}}{\sum_{j=0}^{2} e^{z_j}},$$
computed as `torch.softmax(logits.float(), dim=-1)` (`src/text.py:287`) — **always in fp32, even
when the model itself runs in fp16** on GPU, "for stable probabilities" (`src/text.py:270`). This
matters because summing fp16 exponentials close to the softmax denominator can lose precision;
doing the division in fp32 keeps the reported probabilities numerically well-behaved regardless of
the model's own compute precision.

### Pinned revision and `use_safetensors=False`

```python
FINBERT_REVISION = '4556d13015211d73dccd3fdd39d39232506f3e43'  # pinned commit (HF hub main @ 2026-09-27)
```

I checked this against the Hugging Face Hub directly (read-only API calls, no model loaded):
`4556d13...` is in fact the current HEAD of `ProsusAI/finbert`'s `main` branch, matching the code
comment. Its own commit, however, is dated **2023-05-23** and its message is "Update README.md" —
a documentation-only change. Walking the full commit history (measured via the HF commits API),
the actual model weight file (`pytorch_model.bin` — the format this code loads, since
`use_safetensors=False`) was created in the repo's very first substantive commit, **"First version
of finbert," dated 2020-12-24**, and no commit since has touched that file; the only later
additions were a Flax weights upload (2021-05-18) and a TensorFlow weights upload (2022-06-03) —
different weight *formats*, not changes to the PyTorch weights this code actually loads — plus
several README-only edits, the last of which is the pinned commit itself. So "pinned at
`FINBERT_REVISION`" and "weights last changed 2020-12-24" are both true and consistent: pinning
this specific commit hash guarantees byte-identical `pytorch_model.bin` regardless of anything
that happens to the repo's `main` branch after 2026-09-27, and that file itself hasn't changed
since the day the model was published.

`use_safetensors=False` is a second, distinct guard on top of the revision pin. The comment
explains why it's needed at all: "Without it, transformers silently prefers a safetensors copy
from `refs/pr/29` if one exists, which defeats the pin" (`src/text.py:249-250`). This reflects real
`transformers` library behavior — even with a revision pinned, the loader can auto-fetch a
converted-to-safetensors copy of the weights from an *unrelated* ref (a pull-request branch, not
`main`) if the library's own conversion bot has produced one, unless explicitly told not to. Both
guards together pin the exact bytes being loaded: the revision pin fixes *which commit*, and
`use_safetensors=False` fixes *which file format within that commit's loading logic* — closing off
a code path that could otherwise substitute in weights from somewhere the revision pin doesn't
control.

### The model-side look-ahead argument

The brief is pointed about a specific failure mode for LLM-based signals (p.17): *"A frontier model
trained on data through 2026 already 'knows' what happened to a stock in 2021; asking it for a
view on that period is not forecasting, it is recall... Teams that cannot explain how they ruled
out model-side look-ahead will be treated as having used it."* SPEC amendment A7
(`docs/SPEC.md:104`) is the pipeline's direct answer to that requirement, and `src/text.py`'s
module docstring states the argument precisely (`src/text.py:10-17`): FinBERT's three training
sources — BERT's 2018 pretraining corpora, TRC2 (2008–2010), and the Financial PhraseBank (labelled
2014) — all predate 2021, and (per the commit-history check above) the frozen classifier weights
themselves haven't changed since 2020-12-24. A model whose parameters were fixed in December 2020
cannot have memorized anything about a specific 8-K filed in, say, March 2023, or about that
stock's subsequent price move — there is no possible channel for "recall" of post-2020 information,
because nothing post-2020 was ever in its training data or its weights. This is a narrower, cleaner
guarantee than could be made for a general frontier LLM (which the brief's warning specifically
targets), *and* it's a different kind of guard than the temporal train/valid/test splits used
elsewhere in the pipeline (`models.py`) — those govern what the *learned combination* of features
is allowed to see; this one governs what the *frozen feature extractor itself* could possibly know,
independent of how the pipeline's own splits are set up.

### `MAX_LENGTH` and its history

`MAX_LENGTH = 512` (`src/text.py:47`) truncates the tokenizer input to the first 512 tokens of the
cleaned event body — 512 being BERT's own positional-embedding limit (`config.json`:
`"max_position_embeddings": 512`, matching the code comment `src/text.py:5`), so this is the
*entire* model's usable context, not an arbitrary choice.

This wasn't the first setting tried. Per `docs/research_log.md` (2026-09-27 entry) and
`RUN_FINBERT_DGX.md`: a first CPU-only scoring pass over all 373,139 filings ran at roughly 5–6
docs/second with 10 CPU threads at `MAX_LENGTH=128` (on a laptop also running other jobs),
projecting to about a 20-hour runtime — too slow for a same-day pass
(described as an "event lead": scoring only the first ~128 tokens of the cleaned body, i.e.
essentially the opening sentence or two, not the full event). At 512 tokens, a local CPU smoke run
measured only ~1.1–1.4 docs/s — i.e. days for the full corpus — so full-length scoring is GPU-only.
Once GPU scoring on the user's DGX Spark (GB10 Grace Blackwell, 128GB unified memory) became
available, compute was no longer the binding constraint, and the decision was made — explicitly
*"before any GPU result was seen"* (`src/text.py:44` comment; `docs/research_log.md` 2026-09-27)
— to move to `MAX_LENGTH=512` (the full cleaned body, not just a lead) together with fp16 precision
on GPU. The two changes (compute budget, then token budget) are logged as sequential, cleanly
separated decisions: the device change came first and was made on throughput grounds alone, and
the length change followed from *that*, not from any observed change in FinBERT scores.

**On an "int8" path**: an int8 path was in fact benchmarked and rejected, though the history was
not previously logged in this repository. During the original 128-token CPU run, torch dynamic
int8 quantization (`quantize_dynamic` on `nn.Linear`) was benchmarked on CPU and rejected — the
text agent measured `corr(pos-neg)` int8 vs. fp32 = 0.686 on 200 docs. An Opus verifier
re-checked this on 80 filings at `max_length=128` (2 threads): int8 was 1.72x faster (37.4s vs.
64.4s) but tone correlation was only 0.83 (Spearman 0.815), negative-class correlation 0.61,
top-label agreement 98.75%, and mean `|tone|` gap 0.066 against a tone s.d. of 0.131 — so int8
was rejected on precision grounds, and fp16 on GPU replaced it. This is logged, dated 2026-09-27
(retroactively logged 2026-09-28), in `docs/research_log.md`.

### GPU fp16 path and the precision check

On GPU, `_setup` defaults to `batch_size = GPU_BATCH_SIZE = 256` (vs. `BATCH_SIZE = 32` on CPU) and
`dtype = 'fp16'` (`src/text.py:224-235`); `_load_finbert` calls `model.half()` only when
`device == 'cuda' and dtype == 'fp16'` (`src/text.py:256-257`) — the model stays fp32 on CPU
unconditionally.

`check_precision(n, device, ...)` (`src/text.py:398-430`) is the guard before trusting fp16 at
scale: it scores the same `n` cleaned filings (default 500) once in fp32 and once in the device's
"fast" dtype, then reports `corr(pos - neg)` between the two runs' tone scores and the maximum
absolute per-probability difference, excluding any docs `clean_text` reduced to `''` (nothing to
compare there). `RUN_FINBERT_DGX.md` section 3 instructs running this first via
`python -m src.text --check 500 --device cuda` and states the acceptance bar: *"Expect
`corr(pos-neg) fp32 vs fp16 > 0.999`... If it doesn't clear 0.999, stop and report back rather than
scoring the full corpus in fp16."* I did not run this check myself (out of scope — it requires
loading FinBERT and the DGX GPU isn't part of this environment); I report only that the guard
exists, what it measures, and its documented threshold, not any specific correlation value, since
none is recorded in this repository as having actually been run yet.

### Batching, chunking, and resume

Within a scoring pass, `score_texts` sorts texts by length before batching ("less padding waste"),
scores in fixed-size batches, then restores original order via an explicit index map
(`src/text.py:278-289`) — this changes nothing about individual scores, only compute efficiency.

`score_finbert` (`src/text.py:325-395`) is the outer driver: it reads filing metadata sorted by
`filing_date`, splits it into fixed `CHUNK_SIZE = 1000`-row chunks, and writes each chunk's scores
to its own parquet part file under `chunk_dir_for(max_length)` =
`CACHE_DIR/'finbert_chunks_L{max_length}'`. On each invocation it first checks which chunk files
already exist and only re-scores the missing ones (`src/text.py:345-353`) — an interrupted run (or
a deliberately partial one) resumes exactly where it left off rather than re-scoring everything.
`tests/test_text.py::test_score_finbert_chunks_resume_and_consolidate` verifies this directly: it
deletes one of three chunk files, re-runs, and asserts the model is loaded (and scoring performed)
exactly once more — for the missing chunk only.

### Output file and its guards

`scores_path_for(max_length)` (`src/text.py:319-322`) returns
`CACHE_DIR/'finbert_scores_L{max_length}.parquet'` — the "L" suffix bakes the setting into the
filename specifically so "different max_length/cleaning runs never silently mix" (module
docstring, `src/text.py:27-29`; SPEC section 4, `docs/SPEC.md:45`). Concretely, `MAX_LENGTH=512`
produces `finbert_scores_L512.parquet`, and `RUN_FINBERT_DGX.md` section 4 warns explicitly not to
confuse this with any legacy `finbert_scores.parquet` (no `_L512` suffix) left over from the
original CPU/128-token run — that name is "a different, no-longer-current codepath and is never
read by the current `build_text_features()`."

Consolidation itself has its own guard: the chunk parts are only concatenated into the final
`scores_path_for(max_length)` file once *every* chunk for that exact setting exists
(`src/text.py:388-395`, "written only once every chunk of THIS run's setting exists"); a partial
run prints how many chunks are done and simply doesn't produce (or overwrite) the consolidated
file.

Downstream, `build_text_features` (`src/text.py:475-488`) re-checks the consolidated file every
time it's used, with three separate assertions before trusting it:
1. **Full coverage**: every `document_id` in the filings table must appear in the scores file
   (`assert not missing`) — an incomplete scoring run can't silently produce a panel with some
   filings' tone missing.
2. **`max_length` match**: every row's recorded `max_length` must equal the module's current
   `MAX_LENGTH` (512) — guards against accidentally reading a file built at a different setting
   that happens to have the expected filename.
3. **`revision` match**: every row's recorded `revision` must equal the current
   `FINBERT_REVISION` — guards against reusing scores computed under a different (e.g.
   unpinned, or since-superseded) model revision.

`tests/test_text.py::test_build_text_features_raises_when_scores_settings_mismatch` and
`test_build_text_features_raises_when_scores_missing_document_ids` cover exactly these three
failure modes on synthetic data.

## 4. Features

`TEXT_FEATURES` (`src/text.py:442-446`) is the exact list of model-facing text columns:

```python
TEXT_FEATURES = (
    ['n_filings']
    + [f'item_{c.replace(".", "_")}' for c in config.KEY_ITEMS]   # 12 item_* count columns
    + ['tone_mean', 'tone_min', 'fb_neg_max']
)
```

`build_text_features()` (`src/text.py:449-510`) computes them, keyed by `(permno, eom)` with
`eom` = month-end of `filing_date`:

- **`n_filings`**: simple filing count per `(permno, eom)` — `filings.groupby(['permno',
  'eom']).size()` (`src/text.py:462`).
- **`item_<x_yy>`**: for each `KEY_ITEMS` code, a count of how many of that stock-month's filings
  carried that item — built by exploding the `items` list column, filtering to `KEY_ITEMS`, and
  pivoting to wide counts (`src/text.py:465-473`), zero-filled for codes that didn't occur.
- **`tone_mean` / `tone_min` / `fb_neg_max`**: only computed if `scores_path_for(MAX_LENGTH)`
  exists. Per filing, `tone = fb_pos - fb_neg` (`src/text.py:491`) — the net sentiment, in
  $[-1, 1]$. Aggregated per `(permno, eom)`:
  $$\text{tone\_mean} = \frac{1}{n}\sum_{i=1}^{n} (\text{fb\_pos}_i - \text{fb\_neg}_i), \qquad
  \text{tone\_min} = \min_i (\text{fb\_pos}_i - \text{fb\_neg}_i), \qquad
  \text{fb\_neg\_max} = \max_i \text{fb\_neg}_i$$
  (`src/text.py:492-496`) — mean tone, worst (most negative) tone, and the single strongest
  negative-probability reading among that stock-month's filings. `tone_min` and `fb_neg_max` exist
  specifically to surface a single bad filing even when averaged in with several neutral ones. A
  degenerate case is handled explicitly: if every filing in a `(permno, eom)` group cleaned to
  `''` (so every one of its FinBERT scores is `NaN`), the group's `tone_mean`/`tone_min`/
  `fb_neg_max` are all-`NaN` after the groupby, and `fillna(0.0)` treats that as neutral rather
  than propagating `NaN` into the panel (`src/text.py:498-505`).

**`has_filing`** is produced alongside `TEXT_FEATURES` (`feat['has_filing'] = 1` for every row that
has at least one filing, `src/text.py:463`) but is **deliberately excluded from `TEXT_FEATURES`
itself** — it's carried as a separate auxiliary column. Section 5 explains why at length.

`add_text_features(panel)` (`src/text.py:513-529`) left-joins `build_text_features()`'s output onto
the full panel on `(permno, eom)` and fills every joined column — the `TEXT_FEATURES` columns *and*
`has_filing` — with 0 for stock-months with no matching row, i.e. no filing that month. The
docstring is explicit that this zero-fill is a deliberate state, not a missing-data placeholder:
"stock-months with no filing get 0 counts, 0 tone and `has_filing = 0` (a state, not missing
data)" (`src/text.py:518-519`). `has_filing` is finally cast to `int8` (`src/text.py:528`).

## 5. Survivorship (A1) in depth

### What was discovered

The 8-K archive is, per readme.md, a **retrospectively assembled** dataset — it was built now,
looking back, not accumulated live as filings happened. That matters because whether a firm filed
an 8-K in a given month is not independent of whether that firm is still around to be observed at
all: a firm that's about to be delisted, acquired, or go bankrupt typically stops filing routine
8-Ks before it disappears from the CRSP/Compustat history entirely, while a firm that survives
keeps filing normally indefinitely. In a dataset built by looking backward from "today," the *mere
presence or absence* of a same-month 8-K statistically encodes information about how close that
stock-month is to the firm's eventual exit — information that, at the actual historical
prediction date, no one could have known (the firm's future stopped-filing/delisting hadn't
happened yet). This is a subtle but real look-ahead channel: it's not that any individual filing's
*text* is impossible to have known at the time (the filing did exist by its `filing_date`); it's
that the pipeline's own *feature-presence pattern* — `has_filing`, and more subtly the average
level of `n_filings`/other counts — would correlate with the future outcome being predicted,
purely as an artifact of how the archive was assembled after the fact.

The key observation behind the measured table (below) is that coverage is only 8.9% even 12+
months *before* exit, versus 59.7% for eventual survivors over that same distant-from-any-exit
kind of horizon — a gap nearly as large as the gap right at exit itself. If the effect were driven
mainly by firms *changing their filing behaviour* as they approach exit, coverage should be close
to the survivor rate far from exit and only drop in the final months. It isn't: the gap is present
even a year or more before exit, which means it is mostly **archive composition**, not filing
behaviour — a retrospectively assembled provider archive built around currently-listed NYSE/Nasdaq
firms under-covers firms that later exit, throughout their history in the panel, not just as they
approach the exit itself.

### The measured coverage-by-time-to-exit table

`filing_coverage_by_exit()` (`src/text.py:532-596`) measures this directly. Its design is careful
about a specific confound: it must distinguish a firm's **true exit** (last appearance anywhere in
the *raw*, unfiltered characteristics data — delisting, acquisition, etc.) from a firm merely
**dropping out of the price/size-filtered universe** temporarily while it continues to trade and
file normally elsewhere in the raw data (`src/text.py:540-546`). Using the universe-filtered panel
alone to define "exit" would misclassify that second case as an exit and dilute the real effect.
It also excludes **right-censored** near-term dropouts: a permno last seen fewer than 12 months
before the raw data's own final month can't be distinguished from one that would have reappeared
just past the data's edge, so those stock-months are excluded from every bucket rather than
guessed at (`src/text.py:548-551`).

The corrected, current measurement (SPEC section 10, A1; `docs/SPEC.md:98`; also
`docs/research_log.md`, 2026-09-27 "A1 correction" entry), same-month 8-K filing coverage over
universe stock-months, bucketed by distance to exit:

| Bucket | Coverage |
|---|---|
| Last panel row (the exit month itself) | 1.3% |
| 1–2 months before exit | 3.3% |
| 3–5 months before exit | 3.6% |
| 6–11 months before exit | 3.9% |
| 12+ months before exit | 8.9% |
| Survivor to panel end | **59.7%** |

The gradient is unambiguous: coverage is lowest exactly at exit and rises monotonically with
distance from it, reaching 59.7% for firms still present at the end of the panel — but that 59.7%
is the *survivors'* rate, not the base filing rate for the universe as a whole; the overall filer
base rate across the whole universe is roughly 44–48% (cf. the 44.2%/48% base-rate figures cited
elsewhere in this chapter and in `docs/research_log.md`'s optimizer-infeasibility diagnosis).
**This table was itself corrected once**: an earlier, "less careful" first pass
(`docs/research_log.md`, same-day entry) had reported "0–2.6% vs 40–58%" — directionally the same
finding, but the current table is the one actually locked into SPEC section 10 and the one
`filing_coverage_by_exit()`'s code and tests (`tests/test_text.py::test_filing_coverage_by_exit_buckets_and_censoring`,
`tests/test_text.py::test_filing_coverage_by_exit_runs`) currently reproduce.

### Why it's a look-ahead channel

If `has_filing` (or a feature strongly driven by it) were fed directly into the model, the model
could learn — correctly, in-sample — that "no 8-K this month" predicts "this firm is close to
disappearing," and use that as a return-predictive signal. But at the actual historical prediction
date, the reason a firm stopped filing hadn't happened yet from the perspective of that date; the
correlation exists only because the *archive itself* was built with the benefit of knowing which
firms eventually exited. That's forward-looking information smuggled in through data construction,
not through any single filing's content — exactly the kind of channel the brief's "no misuse of
forward-looking information" requirement (p.17) is aimed at, even though it doesn't come from an
agent or an LLM's training data at all.

### The three defenses

1. **`has_filing` is not a feature.** It's produced by `build_text_features`/`add_text_features` as
   a plain auxiliary column, explicitly separate from `TEXT_FEATURES`
   (`src/text.py:442-446, 522-524`). `tests/test_text.py::test_add_text_features_zero_fill_and_has_filing`
   asserts `'has_filing' not in text.TEXT_FEATURES` directly, and
   `tests/test_integrity.py::test_feature_hygiene_real` bans it (along with `ticker`,
   `company_name`, `stock_exret`, `ret_exc_lead1m`, `target_month`) from ever appearing in the
   real, full feature set actually used by the model (`tests/test_integrity.py:134-144`).

2. **The text specialist is filer-only, with `Z_text = 0` for non-filers.** Per SPEC section 10
   (A1, `docs/SPEC.md:98`): the text specialist model is trained and predicts on filer rows only,
   and its within-month z-score `Z_text` is computed *among filers*, with non-filer stock-months
   set to exactly 0 rather than being imputed or left out. This is implemented in `src/models.py`:
   `run_all` fits the text specialist on `train.loc[train_filer]` only (`train_filer =
   train["has_filing"].astype(bool)`), `_predict_filer_only` returns `NaN` for every non-filer row
   (the model never sees or predicts them), and `_zscore_by_eom` turns that `NaN` into an exact
   `Z = 0` after the within-month z-score among filers.

3. **Filer-net-neutral portfolio constraint (A10).** Per SPEC sections 6 and 10
   (`docs/SPEC.md:63, 107`), the portfolio optimizer adds a constraint bounding
   $\left|\sum_{i:\ \text{has\_filing}_i=1} w_i\right| \le \text{SECTOR\_TOL}$ — i.e., filers as a
   group can't become a systematically net-long or net-short bloc of the book. SPEC A10 states the
   reason directly: "the filer-only text z-score (0 for non-filers) mechanically pushes filers into
   the signal tails (Opus finding: filer share of top-250 61% vs 48% base under pred_ew)." This is
   implemented in `src/portfolio.py`: `_tol_groups` adds a `'filer'` group on the same relaxation
   ladder/tolerance (`SECTOR_TOL`) as the sector groups, and `backtest` records a `filer_net`
   diagnostic column (`(w * has_filing).sum()`) each month so the constraint's effect is directly
   observable.

### The residual-channel analysis from the audit

`docs/research_log.md`'s 2026-09-27 "`_check_constraints`... infeasibility diagnosis" entry records
a further, more granular measurement consistent with A10's own finding: at a real formation month
(2022-06-30), filer status was unevenly spread across the optimizer's signal-ranked *candidate*
set specifically — "filer share of short candidates was 66.4% vs 43.2% of long candidates (vs
44.2% base universe)". That entry frames this as one contributing cause of a separate, unrelated
optimizer-feasibility bug (candidate-set composition leaving little slack in the sector/filer
exposure bands), not as a restatement of the A1 survivorship leak itself — but it is additional,
independently logged evidence that filer status is not a neutral, evenly distributed feature of
the candidate set, reinforcing why A10's explicit filer-net bound (rather than trusting the
optimizer's other constraints to control it incidentally) was judged necessary.

## 6. Tests

### `tests/test_text.py`

Organized in the same order as this chapter:

- **Cover page / item title / TOC** (`test_clean_text_drops_cover_page_and_masks_names`,
  `test_clean_text_skips_boilerplate_item_title_when_on_sentence_found`,
  `test_clean_text_keeps_title_when_no_on_sentence_nearby`,
  `test_clean_text_skips_table_of_contents_item_mention`,
  `test_clean_text_tolerates_malformed_item_header_spacing`,
  `test_clean_text_fallback_when_no_item_pattern`,
  `test_clean_text_short_text_no_item_pattern_keeps_whole_text`): cover every branch of Steps 1–4
  above on small synthetic filings, including the exact TOC-repeat-detection regression and the
  1,500-character fallback boundary (both the "long enough to drop" and "too short, keep whole"
  sides).
- **Boilerplate stripping** (`test_clean_text_strips_furnished_safe_harbor_sentence`,
  `test_clean_text_strips_forward_looking_statement_sentence`,
  `test_clean_text_keeps_substantive_sentence_with_trailing_incorporated_by_reference_clause`,
  `test_clean_text_strips_not_deemed_filed_sentence_without_the_word_furnished`,
  `test_clean_text_cuts_tail_from_item_901_and_signature_block`): each targets one specific
  regression named in the module's own comments — including the "furnished" vs. "filed"-only
  disclaimer wording and the whole-sentence-deletion bug the incorporated-by-reference clause fix
  addressed.
- **Name/ticker masking**
  (`test_clean_text_ticker_masked_only_in_exchange_context`,
  `test_clean_text_short_name_variant_needs_two_words_or_six_chars`,
  `test_clean_text_short_name_variant_masked_when_long_or_multiword`,
  `test_clean_text_short_variant_is_case_sensitive`): directly test the false-positive-avoidance
  gates (length/word-count, case-sensitivity, exchange-context requirement) discussed in section 2.
- **`score_texts`** (`test_score_texts_scores_empty_text_as_nan`): exercises the real (unmocked)
  empty-text-to-`NaN` path using fake tokenizer/model objects, so it needs no FinBERT weights or
  network access.
- **Feature building** (`test_build_text_features_month_assignment_and_counts`,
  `test_build_text_features_includes_tone_when_scores_present`,
  `test_build_text_features_raises_when_scores_settings_mismatch`,
  `test_build_text_features_raises_when_scores_missing_document_ids`,
  `test_truncation_invariance`, `test_add_text_features_zero_fill_and_has_filing`): verify the
  month-assignment boundary (a filing on the 1st of the next month must not count in the prior
  month), the tone-aggregation arithmetic against a hand-computed expected value, both
  scores-file guard assertions from section 3, truncation invariance on synthetic data, and the
  zero-fill/`has_filing` semantics from section 4.
- **`filing_coverage_by_exit`** (`test_filing_coverage_by_exit_buckets_and_censoring`): a fully
  synthetic, hand-constructed four-permno scenario exercising every bucket, the right-censoring
  exclusion, and the "temporary universe drop-out is not a real exit" distinction from section 5,
  with exact expected bucket counts and coverage values asserted.
- **`score_finbert`** (`test_score_finbert_chunks_resume_and_consolidate`,
  `test_chunk_dir_for_separates_max_lengths`,
  `test_score_finbert_device_max_length_plumbing_cpu`): verify chunking/resume behavior (with the
  model mocked — this file never loads real FinBERT, per its own module docstring), that different
  `max_length` settings write to entirely separate chunk directories and output files, and that
  `device`/`max_length`/`batch_size` plumbing doesn't change the scores themselves on CPU, only
  which file/columns record the setting.
- **Slow, real-data tests** (`test_build_text_features_real_data_shape_and_bounds`,
  `test_filing_coverage_by_exit_runs`): sanity checks against the real filing archive — no
  duplicate `(permno, eom)` keys, `eom` bounds matching the documented data range
  (2015-01-31..2026-08-31), and that `filing_coverage_by_exit()` runs end-to-end on real data with
  well-formed output.

### `tests/test_integrity.py`, text-related parts

This file is SPEC section 8's standalone leakage suite and deliberately overlaps a little with
`test_text.py` rather than re-deriving everything from scratch (module docstring,
`tests/test_integrity.py:1-9`). Two parts are specifically about text:

- **`test_truncation_invariance_text_real`** (`tests/test_integrity.py:102-127`): the real-data
  counterpart to `test_text.py`'s synthetic truncation-invariance test. It truncates the *real* 8-K
  archive to `filing_date <= T` (2021-06-30) and asserts month-T text features computed from the
  truncated archive exactly match those computed from the full archive — i.e., nothing filed after
  T could have changed month T's features, on real data, not just a small synthetic frame.
- **`test_feature_hygiene_real`** (`tests/test_integrity.py:133-144`): asserts, on the real feature
  set, that `has_filing` (along with `ticker`, `company_name`, `stock_exret`, `ret_exc_lead1m`,
  `target_month`) never appears among the columns actually fed to the model — the real-data
  enforcement of defense 1 in section 5, run against `data.feature_columns(panel) |
  text.TEXT_FEATURES` rather than against `TEXT_FEATURES` in isolation.
