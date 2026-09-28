# AlphaBERT team guide

An in-depth, code-verified explainer of the AlphaBERT repo (McGill-FIAM 2026 market-neutral
US equity hackathon entry). Each chapter is owned by the module(s) it documents; cross-check
any claim against the cited `file.py::function`, `docs/SPEC.md`, or `docs/research_log.md` —
this guide summarizes and explains, it is not itself the source of truth.

## Reading order

| # | Chapter | What it covers |
|---|---|---|
| 0 | [`00_overview.md`](00_overview.md) | The competition brief in one page, the strategy in one paragraph + pipeline diagram, how the final design differs from the original plan (and why), a full repo map, a conventions glossary, the amendment log (A1–A14) at a glance, and current status — what has and hasn't been run yet. **Start here.** |
| 1 | `01_data.md` | `src/data.py`: the raw panel, universe filter, within-month rank transform, missing-value flags, market state, external market data (TB3MS, S&P 500). |
| 2 | `02_text.md` | `src/text.py`: 8-K text cleaning, FinBERT scoring (cached, chunked, resumable), text feature construction, the survivorship-leak diagnostic behind amendment A1. |
| 3 | `03_models.md` | `src/models.py`: train/valid/test schedule, baselines, the six specialists (5 characteristic groups + text), the gate, the equal-weight headline blend, OOS R²/IC. |
| 4 | `04_portfolio.md` | `src/portfolio.py`: signal smoothing (EMA), the neutral `cvxpy` optimizer (beta/sector/size/filer constraints, the relaxation ladder), monthly return accounting, calibration, submission-file writing. |
| 5 | `05_evaluation.md` | `src/evaluate.py`: performance/exposure/regime/ablation tables, alpha-beta regression, every required chart. |
| 6 | `06_integrity_and_process.md` | `tests/test_integrity.py` and the amendment process: leakage checks, the shuffled-label test, the disclosed smoke-run timeline, the `_check_constraints` bug story. |
| 7 | `07_running.md` | How to actually run the pipeline: `python MAIN.py` (and its flags), the FinBERT DGX runbook, `python -m pytest`, `build_submission_main.py`. |

## Before you read any chapter

No end-to-end run of `MAIN.py` exists yet under the current code. There are no final
performance numbers (IR, Sharpe, alpha, beta, OOS R², holdings) for this strategy — see
`00_overview.md` §0.7 for exactly what is and isn't done. Do not treat any number outside that
section as a result; treat it as a configuration constant or a disclosed diagnostic from an
old, superseded smoke run.
