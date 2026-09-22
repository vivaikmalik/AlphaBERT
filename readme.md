# Stock Characteristics and 8K Filings Dataset Guide

This package combines company disclosures with a monthly stock characteristics panel. Use the filing text to study disclosed information and the security identifiers to connect it to stock observations. This guide describes the supplied data; the competition rules separately determine the prediction target, evaluation periods, eligible investments, execution assumptions, and permitted external data.

## 1. Package contents

Keep these four files together:

```text
readme.md
8k_20150101_20260831_identified.parquet
chars_final_with_names.parquet
factor_char_list.csv
```

| Dataset | Rows | Columns | File size | Observation unit |
|---|---:|---:|---:|---|
| `8k_20150101_20260831_identified.parquet` | 373,139 | 94 | 358,076,009 bytes (358 MB) | One retained filing observation associated with a security |
| `chars_final_with_names.parquet` | 529,082 | 198 | 412,599,998 bytes (413 MB) | One security and calendar month |

| Coverage | Filings | Characteristics |
|---|---|---|
| Dates actually present | `filing_date`: 2015-01-02 to 2026-08-31 | `date`: 2015-01-30 to 2026-08-28 |
| Calendar month labels | Derived from `filing_date` | `eom`: 2015-01-31 to 2026-08-31 |
| Distinct securities (`permno`) | 3,687 | 6,561 |
| Scope | Verified, linked subset of provider NYSE/Nasdaq 8-K archives | U.S. stock-month panel from January 2015 |

**Timing facts for modeling:**

- `ret_exc_lead1m` contains a future return. It must not be used as an input feature for a prediction made in the row's month.
- Filing timestamps do not establish the exact time a disclosure became public. Every filing has `filing_time_precision = 'date_or_midnight_placeholder'`.

## 2. Security identifiers and join keys

| Field | Meaning and storage | Availability |
|---|---|---|
| `permno` | CRSP security identifier; integer. Preferred common security key. | Both files |
| `permco` | CRSP company identifier; integer. A company may have multiple securities. | Both files |
| `gvkey` | Compustat company identifier; six-character string, including leading zeros. | Both files |
| `iid` | Compustat issue identifier; string. Preserve it as supplied, including nonnumeric values. | Both files |
| `gvkey_iid` | Hyphenated convenience key, e.g. `006066-01`. | Filings only; construct from `gvkey` and `iid` if needed in characteristics |
| `cusip` | Nine-character CUSIP, including the check digit; string. | Filings only |
| `cusip8` | First eight characters of `cusip`; string. | Filings only |
| `cik` | SEC registrant identifier; ten-character string with leading zeros. Identifies an issuer, not a share class. | Filings only |
| `ticker`, `company_name` | Period-supported display labels, subject to the evidence and limits below. | Both files; intentionally missing for many characteristics rows |
| `figi`, `composite_figi`, `share_class_figi` | Original provider FIGI identifiers retained for traceability. | Filings only |
| `id` | Upstream panel identifier; equals `permno` for every row in this snapshot. | Characteristics only |

Keep `gvkey`, `iid`, CUSIPs, and CIKs as strings. Tickers and names can change or be reused; do not use them as permanent join keys. CIK or GVKEY alone can combine different share classes. PERMNO-to-GVKEY/IID relationships can also change over time, so use dated relationships.

The characteristics file is unique on **`(permno, eom)`** and also on **`(gvkey, iid, eom)`**. The filing file is unique on `document_id` and on **`(permno, filing_date, text_sha256)`**. Several distinct filings can exist for the same security on the same day.

Every retained filing's PERMNO–GVKEY/IID pair appears somewhere in the characteristics history. This does **not** guarantee a characteristics row in that filing month: 370,201 filings have a same-month link; 2,699 use a unique relationship observed outside that month; 239 use a relationship spanning a gap in the observed history. Consult `characteristics_link_basis` and `characteristics_link_date`.

## 3. Filing dataset

### Content and processing

The source is Quantillium's 8-K archives for provider exchange codes `UN` (NYSE) and `UW` (Nasdaq). These codes are provider classifications, not independently reconstructed daily exchange histories. The final sample contains **373,054 Form 8-K observations and 85 Form 8-K/A amendments**.

The `text` column contains the provider's full main filing text after HTML/whitespace cleanup, removal of identified wrapper and XBRL metadata artifacts, and targeted form-label repairs. Empty or unusable records were removed. Substantive text was not summarized, stemmed, or stripped of stopwords, and it was unchanged during identifier enrichment. Cover pages, signatures, tables, and repetitive legal text can remain. Whitespace normalization can flatten table layout.

This is not a complete EDGAR submission package: referenced exhibits, attachments, images, and press releases are not guaranteed to be present. A filing may incorporate an exhibit by reference without reproducing its contents. `report_url` and source identifiers are retained for provenance; they are not a guarantee of an independently verified SEC filing URL.

Duplicates were resolved within the same PERMNO, provider filing date, and cleaned text hash. Preference was given to a provider document-ID date matching the report date in the content, followed by mapping evidence and a deterministic tie-break. Equivalent provider IDs are retained in `duplicate_source_document_ids`. Identical text linked to different securities remains separate; it can represent multiple share classes associated with the same issuer filing. Amendments and different text versions are not automatically redundant.

The competition release also excludes 6,092 previously linked filings for 42 securities with no characteristics observations from January 2015 onward. Every retained filing security now appears in the distributed panel. This does not imply a filing exists for every stock-month.

### Main content and date fields

| Field(s) | Interpretation |
|---|---|
| `document_id` | Unique provider document ID. **Not an SEC accession number.** |
| `text` | Cleaned main filing text; Arrow `large_string`. |
| `form_type`, `is_amendment`, `form_type_basis` | Form classification, amendment flag, and evidence for classification. |
| `items` | List of detected 8-K item labels; extraction metadata rather than guaranteed exhaustive annotations. |
| `text_chars`, `word_count`, `text_sha256` | Text length, approximate word count, and SHA-256 of the cleaned text. |
| `filing_date` | Preserved provider filing date. Not independently verified against SEC acceptance records. |
| `content_report_date` | Date of report/earliest event extracted from the filing header. **Not a public availability date.** |
| `content_report_date_raw`, `content_report_date_basis` | Extracted source expression and extraction basis. |
| `days_filing_after_content_report` | Calendar-day difference: `filing_date - content_report_date`. Missing when no report date was extracted. |
| `filing_timestamp_utc`, `filing_time_precision` | Provider timestamp represented in UTC, with a precision flag. Midnight/date placeholders must not be treated as actual release times. |
| `sec_accepted_at_utc`, `accession_number`, `isin` | Entirely null in this snapshot; retained schema placeholders. |
| `period_end_date` | Provider metadata; not a validated availability date or fiscal-period endpoint. |
| `filing_date_raw`, `filing_date_utc_raw`, `provider_timezone`, `added_timestamp_raw` | Original date/time strings and timezone metadata, where supplied. |
| `provider_added_at_utc` | Provider ingestion timestamp where interpretable; not the filing's release time. |
| `retrieved_at_utc`, `processed_at_utc`, `research_processed_at_utc`, `identifiers_processed_at_utc` | Download/processing times, not historical information-availability times. |
| `quality_flags`, `source_quality_flags`, `cleaning_flags` | Lists of parser, source, and cleaning annotations. A retained row may still carry a timing or extraction caveat. |
| `title`, `title_ai`, `sector`, `industry`, `provider_is_listed`, `metadata_json` | Provider descriptive metadata; not certified as historical, point-in-time predictors. `title_ai` is a provider AI title, not original filing text. |

Dates are typed calendar dates; timestamp columns carry UTC timezone information. Missing values are null, not zero or empty-date sentinels. Report dates can precede, coincide with, or occasionally conflict with provider filing dates; retain this distinction when defining event windows.

### Identifier evidence and uncertainty

CUSIPs were assigned from dated CRSP `CUSIP9` records. CRSP's latest header CUSIP was not used as historical evidence. The supplied Compustat CUSIP, ticker, company name, and CIK were constant within each security across the exported dates; they were treated as current header attributes and retained separately as `compustat_header_*`.

All retained rows have nonmissing `permno`, `gvkey`, `iid`, `cusip`, `cik`, `ticker`, and `company_name`. CUSIP check digits and source consistency were checked, and CIK/name evidence was corroborated with SEC issuer metadata. Verification is subject to the temporal precision of the sources:

| Evidence field or flag | Interpretation |
|---|---|
| `cusip_source`, `cusip_asof_date`, `cusip_date_precision` | Source record and temporal precision of the assigned CUSIP. The as-of date is a source snapshot date, not an exact effective date. |
| `cusip_date_precision = 'monthly_snapshot'` | 349,181 rows use dated monthly CRSP evidence. Within-month effective dates are not supplied. |
| `cusip_date_precision = 'continuity_inferred_between_snapshots'` | All 23,958 retained 2026 rows infer continuity because December 2025 CRSP CUSIP and the September 2026 Compustat header agree. Intermediate changes cannot be ruled out from those snapshots alone. |
| `FIGI_unverified_link_uses_WRDS_ticker_and_SEC_issuer` in `identifier_quality_flags` | 898 rows use the alternate ticker/issuer link rather than a verified FIGI match. |
| `filing_CIK_differs_from_Compustat_header_CIK` in `identifier_quality_flags` | 1,081 rows have a corroborated filing issuer different from the current Compustat header, including historical entity changes. |
| `2026_ticker_continuity_inferred` in `identifier_quality_flags` | 72 rows have inferred ticker continuity. |
| `cik_source`, `ticker_source`, `company_name_source`, `identifier_mapping_sources` | Evidence used for the assigned identifiers and labels. |
| `provider_ticker`, `provider_company_name`, `compustat_header_*` | Original provider/current-header labels. They can disagree with historical research labels. |
| `content_company_name`, `historical_crsp_company_name` | Parsed registrant and CRSP name evidence; spelling, abbreviations, and share descriptors can differ. |

Flags are lists and can overlap. Source names, snapshot dates, and processing flags are provenance metadata. They should not be assumed observable to investors at the historical prediction date. SEC-style jurisdiction suffixes in names, such as `/MA/`, are retained labels.

### Filing counts by year

| Year | Rows | Year | Rows |
|---|---:|---|---:|
| 2015 | 23,426 | 2021 | 35,124 |
| 2016 | 24,039 | 2022 | 35,946 |
| 2017 | 25,362 | 2023 | 38,181 |
| 2018 | 26,944 | 2024 | 39,411 |
| 2019 | 28,167 | 2025 | 39,608 |
| 2020 | 32,973 | 2026, January–August | 23,958 |

## 4. Monthly characteristics dataset

The original panel was constructed from WRDS CRSP/Compustat data using a U.S. stock characteristics pipeline associated with the methodology of Jensen, Kelly, and Pedersen (2023), *Is There a Replication Crisis in Finance?* The [authors' factor-data website](https://jkpfactors.com/) provides methodological background. This supplied panel is a separately constructed U.S. extract, not an assertion that it is identical to an official public JKP release.

The competition panel keeps observations with `eom >= 2015-01-01` and exactly **147 selected characteristics plus 51 identifiers and auxiliary columns**. All 147 names in the supplied `factor_char_list.csv` are present, once each. No additional characteristic ratios or signals are included. The retained values, nulls, column types, and row order are unchanged from the source; this release does not recompute or normalize characteristics. Pre-2015 observations are omitted, while upstream historical lookback calculations are preserved.

The file has 529,082 observations, 6,561 distinct PERMNOs, and 6,811 GVKEY–IID pairs. The numerical universe remains broader than the linked filing universe.

### Important columns

| Field(s) | Meaning |
|---|---|
| `date` | Last source return observation date in the stock-month. It can precede calendar month-end. |
| `eom` | Calendar month-end label for monthly alignment. It is not a separate publication timestamp. |
| `ret` | Current stock-month return in USD, stored as a decimal: `0.01` means 1%. It is not a next-month return. |
| `ret_local`, `curcd` | Current return in the indicated local currency, and that currency code. |
| `ret_exc` | Current-month return minus the pipeline's risk-free return. Decimal units. |
| `ret_exc_lead1m` | Next-month excess return attached to the current row. Future outcome information, not a contemporaneous feature. Missing at the terminal month and where a valid next-month outcome is unavailable. |
| `ret_exc_wins` | Processed excess return: CRSP values are unchanged; Compustat values are clipped using the pipeline's period-specific 0.1%/99.9% return cutoffs. |
| `source_crsp` | Return-data source: `1` = CRSP, `0` = Compustat. All 2026 rows use Compustat. |
| `prc`, `prc_local` | Price per share in USD and local currency, respectively. |
| `shares` | Shares outstanding in millions. |
| `me`, `me_company` | Security and company market equity, respectively, in USD millions. |
| `me_lag1` | Prior calendar month's security market equity; missing across a month gap. |
| `adjfct` | Share adjustment factor. It uses a common split-adjustment basis across data sources; it is not a return or an independent predictor of future splits. |
| `size_grp` | Size category based on NYSE market-equity breakpoints. |
| `gics`, `sic`, `naics`, `ff49` | Industry classification codes. |
| `common`, `primary_sec`, `exch_main`, `obs_main`, `excntry` | Upstream security/universe classification fields. |
| `primaryexch`, `conditionaltype`, `shrcd`, `exchcd`, `crsp_shrcd`, `crsp_exchcd`, `comp_tpci`, `comp_exchg` | Source-specific security and exchange descriptors; their code systems are not interchangeable. |

### Selected characteristics

Use the `variable` column in `factor_char_list.csv` as the baseline feature-selection list. It contains the following 147 case-sensitive column names:

```text
age aliq_at aliq_mat ami_126d at_be
at_gr1 at_me at_turnover be_gr1a be_me
beta_60m beta_dimson_21d betabab_1260d betadown_252d bev_mev
bidaskhl_21d capex_abn capx_gr1 capx_gr2 capx_gr3
cash_at chcsho_12m coa_gr1a col_gr1a cop_at
cop_atl1 corr_1260d coskew_21d cowc_gr1a dbnetis_at
debt_gr3 debt_me dgp_dsale div12m_me dolvol_126d
dolvol_var_126d dsale_dinv dsale_drec dsale_dsga earnings_variability
ebit_bev ebit_sale ebitda_mev emp_gr1 eq_dur
eqnetis_at eqnpo_12m eqnpo_me eqpo_me f_score
fcf_me fnl_gr1a gp_at gp_atl1 intrinsic_value
inv_gr1 inv_gr1a iskew_capm_21d iskew_ff3_21d iskew_hxz4_21d
ivol_capm_21d ivol_capm_252d ivol_ff3_21d ivol_hxz4_21d kz_index
lnoa_gr1a lti_gr1a market_equity mispricing_mgmt mispricing_perf
ncoa_gr1a ncol_gr1a netdebt_me netis_at nfna_gr1a
ni_ar1 ni_be ni_inc8q ni_ivol ni_me
niq_at niq_at_chg1 niq_be niq_be_chg1 niq_su
nncoa_gr1a noa_at noa_gr1a o_score oaccruals_at
oaccruals_ni ocf_at ocf_at_chg1 ocf_me ocfq_saleq_std
op_at op_atl1 ope_be ope_bel1 opex_at
pi_nix ppeinv_gr1a prc prc_highprc_252d qmj
qmj_growth qmj_prof qmj_safety rd_me rd_sale
rd5_at resff3_12_1 resff3_6_1 ret_1_0 ret_12_1
ret_12_7 ret_3_1 ret_6_1 ret_60_12 ret_9_1
rmax1_21d rmax5_21d rmax5_rvol_21d rskew_21d rvol_21d
sale_bev sale_emp_gr1 sale_gr1 sale_gr3 sale_me
saleq_gr1 saleq_su seas_1_1an seas_1_1na seas_2_5an
seas_2_5na sti_gr1a taccruals_at taccruals_ni tangibility
tax_gr1a turnover_126d turnover_var_126d z_score zero_trades_126d
zero_trades_21d zero_trades_252d
```

The selection includes valuation (`be_me`, `ni_me`), profitability (`niq_be`, `gp_at`), investment (`at_gr1`, `capx_gr1`), return history (`ret_12_1`, `ret_6_1`), risk and liquidity (`beta_60m`, `ami_126d`), and composite measures (`f_score`, `qmj`). `prc` and `market_equity` are part of the selected 147; they are not counted a second time as auxiliary variables. Units and constructions differ across characteristics.

The other 51 retained columns are explicitly:

```text
id permno permco gvkey iid
excntry exch_main common primary_sec bidask
primaryexch conditionaltype crsp_nyse shrcd exchcd
crsp_shrcd crsp_exchcd comp_tpci comp_exchg curcd
fx date eom adjfct shares
me me_company prc_local prc_high prc_low
dolvol tvol ret ret_local ret_exc
ret_lag_dif source_crsp ret_exc_lead1m obs_main gics
sic naics ff49 size_grp ret_exc_wins
me_lag1 ticker company_name ticker_name_reference_date ticker_name_source
ticker_name_status
```

These support identification, time alignment, outcomes, prices and trading volume, market capitalization, currency and share adjustments, industry/universe classifications, and ticker/name provenance. In particular, `ret_exc_lead1m` is an outcome, not an extra predictor. Fields such as `assets`, `sales`, `book_equity`, and `net_income` that were not in the selection list or auxiliary set have been removed.

### Added ticker and company-name columns

| Added field | Meaning |
|---|---|
| `ticker` | Period-supported trading symbol; otherwise null. |
| `company_name` | Period-supported company name; otherwise null. |
| `ticker_name_reference_date` | Date of the supporting identifier or filing observation. |
| `ticker_name_source` | Source of the added label pair. |
| `ticker_name_status` | Verification or missingness status below. |

| `ticker_name_status` | Rows | Treatment |
|---|---:|---|
| `verified_in_observation_month` | 498,706 | Dated CRSP labels from the same month, 2015–2025. |
| `verified_in_filing_on_or_before_observation_date` | 15,731 | 2026 label explicitly corroborated by a retained same-month filing dated on or before `date`. |
| `no_verified_label_for_observation_period` | 14,645 | Labels left null because sufficient period evidence was unavailable. |

There are **514,437 rows with labels and 14,645 without labels**. Missing names do not mean the security IDs or numeric observations are invalid. Historical labels were deliberately not backfilled from later names. In 2026, label availability depends on the retained filing sample; its missingness is therefore not an independent firm characteristic.

Numeric missing values likewise do not mean zero. They can reflect unavailable source data, an undefined ratio, insufficient estimation history, or an unavailable future outcome. Any imputation, scaling, or feature selection should follow the competition's training/evaluation design.

## 5. Reading and joining the files

The following examples use Python, PyArrow, and DuckDB:

```bash
python -m pip install pyarrow duckdb
```

Save the Python code beside the two Parquet files. In a notebook or interactive console, first set the working directory to the folder containing them. This reads metadata and a two-row text preview without loading the full datasets:

```python
from pathlib import Path
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
FILINGS = BASE / "8k_20150101_20260831_identified.parquet"
CHARS = BASE / "chars_final_with_names.parquet"

for path in (FILINGS, CHARS):
    pf = pq.ParquetFile(path)
    print(path.name, pf.metadata.num_rows, len(pf.schema_arrow))
    # Full names and types, without reading the data:
    # print(pf.schema_arrow)

batch = next(pq.ParquetFile(FILINGS).iter_batches(
    batch_size=2,
    columns=["document_id", "filing_date", "permno", "gvkey", "iid", "text"],
))
for row in batch.to_pylist():
    row["text"] = row["text"][:200] + " ..."  # display only
    print(row)
```

For a **descriptive same-month join**, use PERMNO and calendar month-end, and cross-check GVKEY/IID. This example returns five IBM observations and projects only a few columns:

```python
import duckdb

con = duckdb.connect()
con.execute("SET memory_limit = '2GB'")
con.execute("SET threads = 4")
con.read_parquet(str(FILINGS)).create_view("filings")
con.read_parquet(str(CHARS)).create_view("chars")

joined = con.execute("""
    SELECT f.document_id, f.filing_date, f.permno, f.gvkey, f.iid,
           f.ticker AS filing_ticker,
           c.date AS characteristics_date, c.eom, c.me,
           (c.eom IS NOT NULL) AS matched_same_month
    FROM filings AS f
    LEFT JOIN chars AS c
      ON f.permno = c.permno
     AND f.gvkey = c.gvkey
     AND f.iid = c.iid
     AND last_day(f.filing_date) = c.eom
    WHERE f.permno = 12490
      AND f.filing_date >= DATE '2015-01-01'
      AND f.filing_date < DATE '2016-01-01'
    ORDER BY f.filing_date, f.document_id
    LIMIT 5
""").fetch_arrow_table()
print(joined.to_pylist())
con.close()
```

This join is many filings to at most one characteristics row. **It is not an availability-safe feature join:** a filing on 7 January can join a characteristics row based on 30 January data. For prediction, choose the latest characteristics that satisfy the actual information cutoff and the competition's lag policy. Merely matching months, or selecting a prior observation date, does not independently verify the release date of each underlying accounting value.

For stock-month predictions, aggregate eligible filing signals to the chosen stock-month cutoff before joining; otherwise repeated filings can unintentionally duplicate the panel's returns and weights. A left join preserves filings with no same-month characteristic observation. Use an inner join only if that additional sample restriction is intended. Do not assume rows are sorted; sort explicitly for lags and time-series operations.

### Target-month alignment for the competition

For a characteristics row in month `t`, `ret_exc_lead1m` is the excess-return outcome in month `t+1`. It has already been led; do not shift it again. Keep a separate target-month key and assign training, validation, and testing periods by that key. A January 2021 return forecast therefore uses December 2020 characteristics. The first target month available from this January 2015 panel is February 2015.

The competition evaluates holding months January 2021 through August 2026. The legacy teaching scripts use the name `stock_exret` for the target and interpret their `date`, `year`, and `month` as the holding month. Those names are not additional fields in this Parquet file: construct them in a separate model table, while preserving `eom` and the original source `date`. Do not use current-month `ret` or `ret_exc` as if they were a forward target. Select baseline predictors from the CSV list rather than taking every numeric column.

## 6. Actual row previews

Selected fields from two filing rows:

| Filing date | Report date in content | Ticker | PERMNO | GVKEY–IID | CUSIP9 | CIK |
|---|---|---|---:|---|---|---|
| 2015-01-07 | 2015-01-01 | IBM | 12490 | `006066-01` | `459200101` | `0000051143` |
| 2015-02-06 | 2015-02-06 | BWS | 10866 | `002436-01` | `115736100` | `0000014707` |

The IBM row has 2,648 text characters and discusses an officer appointment. The BWS row has 8,961 characters; its research name is `BROWN SHOE CO INC`, while the provider ticker is the later label `CAL`. These examples illustrate why report date, filing date, and current provider labels are kept separately.

Selected characteristics rows for PERMNO 12490 / GVKEY–IID `006066-01` (returns rounded for display):

| `date` | `eom` | `ticker` | `me` (USD millions) | `ret` | `ret_exc_lead1m` |
|---|---|---|---:|---:|---:|
| 2015-01-30 | 2015-01-31 | IBM | 151,930.2100 | -0.044440 | 0.063726 |
| 2026-08-28 | 2026-08-31 | IBM | 221,957.3491 | 0.060920 | null |

The missing future return in the terminal observation must not be replaced with zero.

## 7. Research interpretation and checks

- **Availability:** Content report dates describe reported events. Provider filing dates are unverified event clocks, and processing timestamps describe preparation of this package. Exact intraday event studies require additional SEC acceptance-time evidence.
- **Future information:** Separate outcomes such as `ret_exc_lead1m` from predictors. A current month's returns, prices, and month-end characteristics are not available at the beginning of that month. Identical or related filings across securities can also contaminate a randomly split text-learning exercise; respect the competition's time and grouping rules.
- **Selection:** Filing retention and identifier matching use retrospectively assembled histories, including 2026 source exports. The package is not a reconstruction of the investable universe known at each historical date.
- **Labels and provenance:** Display labels, current provider/header metadata, AI titles, and audit flags should not be treated as historical signals without a separate availability rationale. Missing 2026 names partly reflect the way this package was enriched.
- **Audit scope:** Retained identifiers and duplicate keys were checked; cleaned filing text was verified against the pre-enrichment text; all retained characteristics values and types were compared with the source. The 2015 date filter, exact 147-characteristic selection, auxiliary set, unique stock-month keys, and retained filing-to-panel membership were also checked. These checks do not certify complete SEC coverage, daily identifier effective dates, original accounting vintages, or a particular trading simulation.

### File integrity

These SHA-256 checksums identify the exact distributed dataset snapshot. Rewriting a Parquet file, even with identical logical rows, will normally change its checksum.

```text
8k_20150101_20260831_identified.parquet
f333cd02076485ec3380e73d78c5d9bcc689b30827e7332ed6f699f58ab0cd07

chars_final_with_names.parquet
797c01ebd59f7a14585cb2e0290aa0b2c43b9188b7ac16384762d3df615215ca
```

The feature-list CSV is included for reproducible column selection. The schemas and provenance columns are embedded in the two files. No download scripts, API credentials, source CSVs, or intermediate audit files are needed to read this package.
