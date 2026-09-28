# 7. Running the pipeline

This chapter is the runbook: what you need before you type anything, how to sanity-check on a
laptop, the copy-pasteable end-to-end sequence for the DGX Spark, what `MAIN.py` actually does
and prints while it runs, the full output-file reference, and troubleshooting. Every command,
flag and path below was checked against the current code (`MAIN.py`, `src/config.py`,
`src/text.py`, `src/data.py`, `src/evaluate.py`, `src/portfolio.py`,
`build_submission_main.py`) — not against `docs/SPEC.md` prose, which can drift.

## 7.1 What you need

### Files that are not in git

`.gitignore` excludes `data/`, `Data/`, every `*.parquet`/`*.feather`/`*.h5`/`*.hdf5`, and all of
`outputs/`. A `git clone` of the branch gets you code only — none of this exists on a fresh
checkout:

| Path | Used by | Notes |
|---|---|---|
| `data/chars_final_with_names.parquet` | `src/config.CHARS_PATH` | ~394 MB. Raw characteristics panel. |
| `data/8k_20150101_20260831_identified.parquet` | `src/config.FILINGS_PATH` | ~342 MB. 8-K filing text + metadata, 373,139 rows. **Competition-provider data — see licensing note below.** |
| `data/factor_char_list.csv` | `src/config.CHAR_LIST_PATH` | Small. List of the 147 characteristic names; `config.load_char_list()` asserts there are exactly 147. |
| `data/external/tb3ms.csv`, `data/external/sp500.csv`, `data/external/SOURCES.md` | `src/data.load_market()` | Small. If missing, `load_market()` downloads them itself (see internet needs below) and writes `SOURCES.md` describing the source/date. |

Copy them onto whatever machine you're running on. From the laptop (adjust the destination):

```bash
# everything under data/ (394+342 MB parquet + the small csvs)
scp -r "D:/Work/Hackathon/AlphaBERT/data" <user>@<host>:~/alphabert/
# or, if data/ already exists on the target and you just want to sync it:
rsync -avz "D:/Work/Hackathon/AlphaBERT/data/" <user>@<host>:~/alphabert/data/
```

If `data/external/*` isn't copied, `load_market()` will try to fetch it itself the first time
`MAIN.py` or `data.load_market()` runs — see below.

### Internet needs

Two independent things reach out to the network, both only when their local cache/file is
missing:

1. **HF Hub model download.** `src/text._load_finbert()` calls
   `AutoTokenizer.from_pretrained(FINBERT_MODEL, revision=FINBERT_REVISION, use_fast=True)` and
   the matching `AutoModelForSequenceClassification.from_pretrained(..., use_safetensors=False)`,
   `FINBERT_MODEL = 'ProsusAI/finbert'`, pinned at
   `FINBERT_REVISION = '4556d13015211d73dccd3fdd39d39232506f3e43'` (`src/text.py`). This triggers
   on the *first* `--check` or `--score` call (any device) if the revision isn't already in the
   local HF cache (`~/.cache/huggingface` by default, or `$HF_HOME`). If the DGX has no outbound
   internet, pre-fetch the same snapshot on the laptop and copy the cache dir over — see
   `docs/RUN_FINBERT_DGX.md` section 2 for the exact `HF_HOME=... python -c "..."` snippet.
2. **FRED / yfinance**, only if `data/external/tb3ms.csv` or `data/external/sp500.csv` is
   missing: `src/data._download_market_data()` pulls TB3MS from
   `https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS`, and S&P 500 total return from
   yfinance (`^SP500TR`, monthly), falling back to FRED's `SP500` *price* index (no dividends) if
   yfinance fails — the fallback prints a warning and stamps `sp500_source = 'price_only'` in
   `data/external/SOURCES.md`. If you already copied `data/external/` from the laptop, this
   codepath never fires.

Everything else (`data.build_panel()`, `data.market_state()`, `models.run_all()`,
`portfolio.backtest()`, `evaluate.run_evaluation()`) is pure local compute on files already on
disk — no network needed.

### Licensing note

`docs/RUN_FINBERT_DGX.md` already flags this: `data/8k_20150101_20260831_identified.parquet` is
competition-provider data. Keep it (and `data/chars_final_with_names.parquet`) on team machines
only. `.gitignore` already keeps `data/` and every `*.parquet` out of the git tree, so pushing
your branch to a remote — public or private — will **not** upload this data as long as you don't
`git add -f` it. Don't `git add -f` it. `outputs/` is also entirely gitignored, so
`outputs/submission/{holdings,returns,label_audit}.csv` (the actual competition deliverable)
never gets pushed either — copy those back by hand (7.3, step 8).

## 7.2 Laptop run (Windows/Linux)

This is a CPU-only sanity pass — real, but slow (FinBERT scoring at CPU speed is the bottleneck).
`docs/research_log.md`'s ~5-6 docs/s (~20h ETA for the full corpus) measurement was taken at
`MAX_LENGTH=128`; at the current `MAX_LENGTH=512`, a local CPU smoke run measured only
~1.1-1.4 docs/s, which is days for the full 373k-document corpus — full FinBERT scoring is
GPU-only in practice at the current sequence length. Use this laptop pass to confirm the
environment and code are correct before touching the DGX, not to produce a full FinBERT run.

```bash
cd "D:/Work/Hackathon/AlphaBERT"
python -m venv .venv && source .venv/Scripts/activate   # or conda; see requirements.txt
pip install -r requirements.txt

# 1. Cheapest possible check: loads/builds the panel + text features + market data, prints
#    shapes, and exits before any model fit / backtest / evaluate / submission.
python MAIN.py --dry

# 2. Confirm src/text.py's CLI and FinBERT setup work at all (n=500 docs default). This is the
#    thing that triggers the HF Hub download if the model isn't cached yet. On CPU there is no
#    fp16 to compare against, so this is only an fp32-vs-fp32 identity/plumbing check -- it loads
#    the model and scores 500 docs, which takes minutes, not the fp32-vs-fp16 precision check
#    §7.3 step 5 runs on the DGX.
python -m src.text --check

# 3. Full run (recomputes models.run_all() -- no --reuse-preds on a first run, there's nothing to
#    reuse yet).
python MAIN.py
```

`python MAIN.py --dry` is fast (seconds to a couple minutes — it still builds/caches the panel
and reads the FinBERT scores cache if present, just skips modeling). The full `python MAIN.py`
run is dominated by `models.run_all()`: 6 annual refits, each fitting 4 baselines (OLS/ridge/
lasso/elastic-net), one LightGBM-on-everything, 5 characteristic-group LightGBM specialists, and
1 text-LightGBM specialist, plus the ridge gate. An earlier interface smoke run (older, pre-A13
model code) measured `run_all()` at ~48 minutes (2,860s) on this laptop; the current code tunes
up to 1,000 boosting rounds × 2 leaf-count settings per specialist, then refits, so expect this
run to take longer than that 48-minute figure on the same laptop (this will vary with core count
and `N_JOBS`; treat it as a rough planning number, not a guarantee). Full CPU FinBERT scoring (`python -m src.text --score --device cpu`) is not practical
on a laptop at the current `MAX_LENGTH=512` (days, per the ~1.1-1.4 docs/s measurement above) — so
on a laptop run, `MAIN.py` will typically print `tone included: False` and proceed without tone
columns (this is intentional — text features degrade gracefully, see
`src/text.add_text_features`); full scoring is expected to run on the DGX GPU (§7.3 step 6).

Outputs land under `outputs/{cache,tables,figures,submission}/` — see 7.5 for the full list.

## 7.3 DGX Spark end-to-end

Copy-pasteable sequence for the DGX Spark (GB10 Grace Blackwell, ARM64/aarch64, CUDA GPU, 128 GB
unified memory), run from inside an NGC PyTorch container. Replace `<branch>`, `<repo-url>`,
`<user>@<dgx-host>` and `<tag>` as needed.

**0. Push your branch, then clone it on the DGX.**

```bash
# on the laptop
git push -u origin <branch>
```

```bash
# on the DGX
git clone -b <branch> <repo-url> ~/alphabert
```

**1. Copy the data over** (not in git — see 7.1):

```bash
# from the laptop
scp -r "D:/Work/Hackathon/AlphaBERT/data" <user>@<dgx-host>:~/alphabert/
```

A fresh `git clone` has no `outputs/` directory at all (it's entirely gitignored, per 7.1) — so on
the DGX, the panel, text features, and predictions all get rebuilt from `data/` from scratch on
the first run; there is no stale cache to worry about clearing.

**2. Start the container**, mounting the repo:

```bash
ssh <user>@<dgx-host>
cd ~/alphabert
docker run --gpus all -it --rm -v $PWD:/work -w /work \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:<tag>-py3 bash
```

The HF cache mount matters because the container is started with `--rm`: without it, the FinBERT
model snapshot downloaded into the container's own `/root/.cache/huggingface` is lost when the
container exits, and the next run re-downloads it from scratch.

Pick `<tag>` from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags (an aarch64
build with CUDA support, e.g. `25.08`); the exact version isn't load-bearing.

**3. Install the remaining Python packages, skipping torch.** The container already ships a
CUDA-enabled aarch64 PyTorch build — `pip install`-ing `torch` from `requirements.txt` risks
pulling a generic (non-CUDA, or x86) wheel over the container's own build. Filter it out:

```bash
grep -v '^torch$' requirements.txt > /tmp/requirements-dgx.txt
pip install -r /tmp/requirements-dgx.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # sanity check
```

`requirements.txt` pins no versions, so this pulls latest-compatible for every package. Most
(`pandas`, `numpy`, `pyarrow`, `scikit-learn`, `statsmodels`, `duckdb`, `yfinance`, `scipy`,
`transformers`) are pure Python or ship aarch64 wheels routinely. `lightgbm`, `cvxpy` and its
solver backends `clarabel`/`scs` (both pulled in as `cvxpy` extras/deps) also publish aarch64
manylinux wheels on recent releases. If any of them instead falls back to a source build and
fails:
- missing compiler/cmake: `apt-get update && apt-get install -y build-essential cmake` (NGC
  containers usually already have these, but confirm if `lightgbm`'s build step complains).
- pip trying to build in an isolated env without network access to fetch its own build deps:
  add `--no-build-isolation` to the failing package's `pip install` (only if you've separately
  confirmed the build tools are present).
- as a last resort, `conda`/`mamba install -c conda-forge <pkg>` inside the container if the base
  image has conda; NGC PyTorch images don't by default, so this needs conda installed first.

**4. Raise `N_JOBS`.** `src/config.py` has `N_JOBS = 6` (line 52) as a plain module constant —
there's no CLI flag or env var for it. It's the only thing that scales with CPU cores (it flows
through to LightGBM's `n_jobs` via `src/models.py`'s `_lgbm_base_params` — `fit_lgbm` no longer
exists as a separate function). The DGX Spark has 20 ARM cores; edit the constant before running
anything CPU-bound:

```bash
sed -i "s/^N_JOBS = 6$/N_JOBS = 16/" src/config.py
```

(Leaving a few cores free rather than setting it to 20 is a reasonable default; there's no rule
against 20, it's just not free lunch once you're also running the GPU FinBERT pass concurrently.)

**5. FinBERT precision check** (do this before scoring the full corpus):

```bash
python -m src.text --check 500 --device cuda
```

Expect `corr(pos-neg) fp32 vs fp16 > 0.999` in the printed output (`src/text.check_precision`).
If it doesn't clear that bar, stop — don't run the full `--score` in fp16.

**6. Full FinBERT scoring pass.** `--device` defaults to `'cpu'` in `src/text.py`'s argparse — it
is *not* auto-detected, so it must be passed explicitly for GPU:

```bash
python -m src.text --score --device cuda
```

(`--max-length` defaults to `MAX_LENGTH = 512` already, so it doesn't need to be repeated; add it
only if you deliberately want a different setting.) This is resumable — chunks already written
under `outputs/cache/finbert_chunks_L512/` are skipped on re-run. Expected runtime on the DGX GPU
is unmeasured as of this writing; expect tens of minutes rather than the "well under 15 minutes"
figure earlier documentation suggested, since scoring 373,139 filings also includes CPU-side text
cleaning of all of them, not GPU inference alone. Output:
`outputs/cache/finbert_scores_L512.parquet` (373,139 rows), written once every chunk exists.

**7. Verify the scores file:**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('outputs/cache/finbert_scores_L512.parquet')
assert len(df) == 373139, len(df)
assert df['document_id'].is_unique
probs = df[['fb_pos','fb_neg','fb_neu']]
finite = probs.notna().all(axis=1)
assert (probs[finite].sum(axis=1).sub(1.0).abs() < 1e-4).all()
assert (df['max_length'] == 512).all()
print('OK:', len(df), 'rows,', (~finite).sum(), 'empty-text (NaN) rows excluded')
"
```

A handful of NaN rows (filings `clean_text()` stripped to nothing) is expected and fine.

**8. Confirm tone is now included, then run the full pipeline:**

```bash
python MAIN.py --dry
# look for: "tone included: True" in the printed output
python MAIN.py
python build_submission_main.py
python -m pytest -q
```

`python -m pytest -q` runs the whole suite, including everything marked `@pytest.mark.slow`
(`tests/conftest.py` registers the marker but nothing in `pytest.ini` deselects it, so `-q` alone
does not skip slow tests) — several of them fit real LightGBM models on a subset of the real
panel or read the real 8-K corpus, so expect a few minutes, not seconds. To skip them for a quick
check: `python -m pytest -q -m "not slow"`.

**9. Copy results back to the laptop:**

```bash
# from the laptop
scp -r <user>@<dgx-host>:~/alphabert/outputs "D:/Work/Hackathon/AlphaBERT/"
```

(`outputs/` is gitignored on both ends, so this scp is the only way the results get back — see
the licensing note in 7.1.)

## 7.4 What `MAIN.py` does, step by step

`MAIN.py` prints a banner for each of 8 steps (`=== step N/8: ... ===`) plus a final
`=== headline ===` dict. In order:

1. **`build panel, market state, market data`** — `data.build_panel()` (reads/caches
   `outputs/cache/panel.parquet`; prints `universe rows: N of M raw rows (K months)` and the
   selected `miss_` flags), `data.market_state()` (recomputed every call, not cached — it's
   cheap), `data.load_market()` (downloads to `data/external/` only if missing, per 7.1).
2. **`text features`** — `text.add_text_features(panel)` left-joins `TEXT_FEATURES`
   (`n_filings`, 12 `item_*` counts, and, only if the FinBERT scores cache is complete,
   `tone_mean`/`tone_min`/`fb_neg_max`). Prints whether
   `outputs/cache/finbert_scores_L512.parquet` exists and whether tone was actually included, and
   the list of text feature columns in use. **`--dry` stops here**, after printing panel/
   market_state/market shapes.
3. **`models.run_all (or reuse cached preds)`** — without `--reuse-preds`, fits everything fresh:
   for each of the 6 test years (2021..2026) it prints
   `[models.run_all] test_year=Y train=... valid=... (labelled=...) test=... (Xs)`, fitting 4
   linear baselines, 1 "lgbm on everything" model, 5 characteristic-group specialists
   (`pred_spec_value/momentum/quality/investment_growth/risk_liquidity`), 1 text specialist
   (`pred_spec_text`, filer rows only), the ridge gate, and the equal-weight blend `pred_ew`/
   `pred_ew_ret`. Writes `outputs/cache/preds.parquet` and `outputs/cache/gate_coefs.parquet`,
   prints the OOS R2/IC table (also written to `outputs/tables/oos_r2.csv`) and the headline
   `pred_ew_ret` OOS R2.
4. **`calibrate portfolio penalties`** — grid-searches `L2_PENALTY`/`TURNOVER_PENALTY` (16
   combinations) on the smoothed 2019-2020 validation `pred_ew` signal only, by re-running
   `portfolio.backtest()` itself (`portfolio.calibrate`), scored on portfolio shape (names/side,
   turnover), never on realized returns. Prints the locked `L2_PENALTY`/`TURNOVER_PENALTY` and
   the full grid, writes `outputs/tables/calibration.csv`.
5. **`headline backtest`** — smooths `pred_ew` (`portfolio.smooth`: within-eom z-score, then a
   per-permno EMA with `EMA_ALPHA=0.5`, reset on any skipped month) and runs
   `portfolio.backtest()` with the locked penalties over formation months 2020-12..2026-07 (68
   holding months, 2021-01..2026-08). Attaches ticker/company-name labels
   (`portfolio.attach_labels`) and computes the adverse missing-return sensitivity check
   (`portfolio.missing_return_sensitivity`).
6. **`ablations`** — reruns the *identical* smoothing + locked penalties + `portfolio.backtest()`
   for 11 more signals: `pred_gate`, `pred_gate_notext`, `pred_ew_notext`, `pred_lgbm_all`,
   `pred_ridge`, and all 6 `pred_spec_*` columns. This means **12 full optimizer backtests total**
   (headline + 11 ablations), each solving a constrained QP per formation month via `cvxpy`
   (CLARABEL, falling back to SCS). This is *not* likely to be the wall-clock bottleneck: each
   backtest is 68 solves at roughly 0.1-0.5s each, and step 4's calibration is 16×24 solves — all
   on the order of minutes total, well under `models.run_all` (step 3), which dominates the
   pipeline's runtime. Prints `ablation done: <col>` per signal.
7. **`evaluate`** — `evaluate.run_evaluation()` writes every table in `outputs/tables/` and every
   chart in `outputs/figures/` (full list in 7.5), returns the headline stats dict.
8. **`write submission`** — `portfolio.write_submission(holdings, returns)` writes the 3-file
   submission to `outputs/submission/` (7.5).

**`--reuse-preds`**: skips step 3's fit and loads `outputs/cache/preds.parquet` +
`gate_coefs.parquet` instead. Use it when you've already had a *successful* `run_all()` (step 3
completed and wrote both cache files) and only want to re-run portfolio construction/evaluation —
e.g. after tweaking `src/portfolio.py` or `src/evaluate.py`, not `src/models.py`. It has a
built-in staleness guard (`MAIN.py`, around the `preds_path` check): it raises `RuntimeError` if
`preds.parquet` is missing any column `models.PRED_COLS` now expects, or if `preds.parquet` is
*older* than `src/models.py` itself (by mtime) — either case means the cache predates the current
model code, and it tells you to re-run without `--reuse-preds` rather than silently using a stale
cache.

## 7.5 Every output file

### `outputs/cache/`

| File | Written by | Contents |
|---|---|---|
| `panel.parquet` | `data.build_panel()` | The universe-filtered, ranked panel (permno/eom keyed): 147 within-eom-ranked characteristics, `miss_*` flags, `gics2`/`beta`/`size_z`/aux columns, `stock_exret`/`target_month`. |
| `preds.parquet` | `models.run_all()` | One row per (permno, eom, test_year, split), all `PRED_COLS` (the 4 baselines, `pred_lgbm_all`, 6 `pred_spec_*`, `pred_gate`, `pred_gate_notext`, `pred_ew`, `pred_ew_ret`, `pred_ew_notext`). |
| `gate_coefs.parquet` | `models.run_all()` | Ridge gate coefficients per (test_year, specialist, term); `term` is `'base'` or a `config.STATE_VARS` name, plus per-test_year state mean/std — consumed by `evaluate.gate_weight_series`/`plot_gate_weights`. |
| `finbert_chunks_L512/` | `text.score_finbert()` | Resumable per-chunk FinBERT output, 1000 filings/chunk (`part_00000.parquet`, ...). |
| `finbert_scores_L512.parquet` | `text.score_finbert()` | Consolidated FinBERT scores, 373,139 rows: `document_id, permno, filing_date, fb_pos, fb_neg, fb_neu, max_length, device, dtype, revision`. Written only once every chunk for this setting exists. |

Note the `_L512` suffix (`text.scores_path_for`/`chunk_dir_for`) — a differently-configured run
(different `max_length`) never overwrites or mixes with this one.

### `outputs/tables/*.csv`

All written by `evaluate.run_evaluation()` except `oos_r2.csv` (step 3) and `calibration.csv`
(step 4), both written directly by `MAIN.py`.

| File | Contents |
|---|---|
| `oos_r2.csv` | `model, oos_r2, oos_r2_demeaned, mean_ic, ic_tstat` per `PRED_COLS` entry (test rows only). |
| `calibration.csv` | The 16-point `L2_PENALTY`×`TURNOVER_PENALTY` grid with `avg_names_per_side`, `avg_turnover`, `score`. |
| `performance_table.csv` | Rows `strategy_gross`, `strategy_net`, `benchmark`, `sp500`, `long_leg_excess_contrib`, `short_leg_excess_contrib`; columns avg/annualized/CAGR/cumulative return, Sharpe, max drawdown, IR, hit rate, best/worst month, `cost_basis`, `corr_sp500`. |
| `alpha_beta.csv` | Rows `gross`/`net`; OLS and Newey-West (3-lag) alpha/beta vs `sp500_exret`, SEs, t-stats, `n_obs`. |
| `calendar_year_table.csv` | Compounded strategy (gross/net)/benchmark/S&P 500 return by calendar year (2026 row labeled "YTD (Jan–Aug)"). |
| `exposure_table.csv` | Avg long/short name counts, gross/net exposure, filer-net exposure, tolerance-relaxation shares/counts, avg/max abs weight, top-10 concentration, turnover, ex-ante beta, missing-return weight. |
| `short_book_table.csv` | Short-leg diagnostics: weighted/simple avg market cap and dollar volume, share of weight in nano/micro caps, share in bottom-quintile dollar volume. |
| `contributors.csv` | Top/bottom 10 permnos by summed excess-return contribution (`weight * stock_exret`), with ticker/company_name labels. |
| `top_holdings.csv` | Top 10 long / top 10 short names by average weight across all 68 test months. |
| `regime_table.csv` | IR, annualized active return, long/short contribution, month count, by regime (`2021`, `2022`, `2023-2025`, `2026`). |
| `ablation_table.csv` | IR, Sharpe, annualized active return, beta, turnover, max drawdown per ablation signal (5 fixed + 6 `pred_spec_*`). |
| `sensitivity_table.csv` | Headline vs. adverse-missing-return-fill headline stats, side by side. |

### `outputs/figures/*.png`

All written by `evaluate.run_evaluation()` (`gate_weights.png` only when `gate_coefs`/`state` are
passed — `MAIN.py` always passes both).

| File | Contents |
|---|---|
| `cumulative_returns.png` | **The headline chart.** Cumulative return of the strategy (`total_ret`) vs. the benchmark (`bench_ret` = T-bill rate + 4%/yr hurdle) vs. the S&P 500 (`sp500_ret`), gross of costs, over the test window, anchored at 0 on 2020-12-31. |
| `underwater.png` | Drawdown (strategy vs. S&P 500). |
| `rolling_active_return.png` | 12-month rolling annualized active return vs. the benchmark. |
| `rolling_ir.png` | 12-month rolling information ratio. |
| `rolling_beta.png` | 12-month rolling beta vs. the S&P 500. |
| `return_histogram.png` | Distribution of monthly strategy returns, with the average benchmark hurdle marked. |
| `contributors.png` | Bar chart of the top/bottom 10 return contributors. |
| `gate_weights.png` | Effective ridge-gate specialist weights over the test window. |

### `outputs/submission/`

| File | Written by | Contents |
|---|---|---|
| `holdings.csv` | `portfolio.write_submission()` | `Date, PERMNO, TICKER, COMPANY NAME, WEIGHT` (percent, capped at `MAX_WEIGHT*100`, each month's long leg sums to +100 and short leg to -100 after rounding-residual reallocation). |
| `returns.csv` | `portfolio.write_submission()` | `Date, total_ret, rf_m, bench_ret, active_ret, ls_ret, long_ret, short_ret, sp500_ret, total_ret_net, active_ret_net` — one row per holding month. `total_ret`/`active_ret` are gross of trading costs (the headline); `*_net` are cost-adjusted. |
| `label_audit.csv` | `portfolio.write_submission()` | `permno, month, ticker, company_name, label_source` — `label_source` is `panel`, `raw_panel`, `filing`, or `UNLABELED`, tracing where each name/ticker label came from. |
| `MAIN.py` | `build_submission_main.py` | Single-file bundle: `src/{config,data,text,models,portfolio,evaluate}.py` and `MAIN.py`'s own body concatenated, each module `exec`'d into its own namespace — runs with no `src/` package needed. Paths resolve relative to `$ALPHABERT_ROOT` (fallback: cwd), e.g. `ALPHABERT_ROOT=/path/to/AlphaBERT python outputs/submission/MAIN.py --dry`. |

`build_submission_main.py` has no CLI flags (no argparse at all — it always just runs `build()`
when executed; `python build_submission_main.py --help` runs the build and ignores the
unrecognized argument).

## 7.6 Troubleshooting

- **Stale caches after code changes.** `data.build_panel()` and `models.run_all()`'s
  `--reuse-preds` path both trust an on-disk cache without checking every input. If you changed
  `src/data.py`'s panel-building logic, delete `outputs/cache/panel.parquet` — nothing else does
  it for you. If you changed `src/models.py` and want a clean refit, just don't pass
  `--reuse-preds` (or delete `outputs/cache/preds.parquet` and `gate_coefs.parquet`) — recall
  `--reuse-preds` itself already refuses a `preds.parquet` older than `src/models.py` or missing
  expected columns (7.4), so the main failure mode this guards against is *forgetting the flag
  isn't set*, not a truly stale cache silently being reused.
- **FinBERT scores file missing or mismatched.** `text.build_text_features()` asserts (a) every
  `document_id` in the filings file is present in the scores file, (b) every row's `max_length`
  equals `MAX_LENGTH` (512), and (c) every row's `revision` equals the pinned
  `FINBERT_REVISION`. Any of these failing means `outputs/cache/finbert_scores_L512.parquet` is
  incomplete or was produced by a different setting/pin — re-run `python -m src.text --score`
  to completion at the current settings rather than editing the parquet by hand. If the file
  doesn't exist at all, `build_text_features()` just prints a warning and omits tone columns —
  not an error, but check `MAIN.py`'s step 2 printout (`tone included: ...`) if you expected tone
  to be there.
- **Solver infeasible / "solver failed even after relaxation".** `portfolio.optimize_month`
  raises `RuntimeError("optimize_month: solver failed even after relaxation")` only after trying
  both CLARABEL and SCS at all 4 steps of the tolerance-relaxation ladder
  (`portfolio.RELAX_STEPS`). This should be rare; if it fires, it's most likely a genuinely
  infeasible month (e.g. too few candidate names passing the universe filter, or a signal that's
  NaN/degenerate for that month) rather than a solver flake — check `panel.parquet` for that
  `eom` and confirm `N_CAND=250`-worth of non-null-signal candidates exist per side.
- **Memory.** The two data files are ~394 MB and ~342 MB — trivial against the DGX's 128 GB
  unified memory or a laptop with 16+ GB. `text._texts_for_ids` loads the full 8-K text column
  into memory as raw strings (~1 GB per its own docstring), which is fine anywhere this pipeline
  is meant to run. If you do hit memory pressure, it's more likely GPU batch size
  (`--batch-size`, default 256 on CUDA) during FinBERT scoring than host RAM — lower it rather
  than reworking the pipeline.
- **pytest slow.** `python -m pytest -q` runs `@pytest.mark.slow` tests by default (the marker is
  registered in `tests/conftest.py` but nothing deselects it) — several read the real parquet
  files or fit real LightGBM models on a panel subset (`tests/test_models.py`,
  `tests/test_text.py`, `tests/test_integrity.py`, `tests/test_data.py`). Use
  `python -m pytest -q -m "not slow"` for a fast pass, `python -m pytest -q -m slow` to run only
  the real-data tests.
