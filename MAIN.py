"""AlphaBERT -- end-to-end pipeline entry point (owns this file only; see docs/SPEC.md section 9/10).

    python MAIN.py                  # full run: recompute predictions, then backtest/evaluate/submit
    python MAIN.py --reuse-preds    # reuse outputs/cache/{preds,gate_coefs}.parquet from a prior run
    python MAIN.py --dry            # load cached panel/text/market, print shapes, exit (no fit/backtest)

Cache-aware throughout (src/data.py, src/text.py): an empty outputs/cache/ reproduces the whole
pipeline from the raw parquet files in data/.

Steps (each a thin call into the owning src module -- no modeling/portfolio logic lives here):
  1. Build the panel, market-state series and external market data (T-bill, S&P 500).
  2. Attach text (8-K) features; tone columns are included only when the settings-specific FinBERT
     score cache (text.scores_path_for(text.MAX_LENGTH)) is a complete run.
  3. models.run_all(): baselines, the five A9 characteristic specialists + text, and the ridge
     gate, per test year -> OOS R2/IC table. The headline's R2 comes from config.HEADLINE_SIGNAL:
     when it's pred_ew (A12), pred_ew_ret (its return-unit twin) is reported; when it's pred_gate
     (pre-registered, A16), pred_gate is itself return-unit and already has an R2 in the table.
  4. Calibrate the optimizer's two penalties ONCE on the SMOOTHED 2019-2020 validation pred_ew
     signal (A3, amended: smoothed via portfolio.smooth, not raw), then lock them.
  5. Headline backtest: config.HEADLINE_SIGNAL (pred_gate, the pre-registered ridge regime gate,
     per A16 -- see docs/research_log.md; A12's pred_ew was a post-hoc headline switch and is now
     reported only as an ablation), the neutral optimizer (A14 sector-demeaned candidates) over
     formation months 2020-12..2026-07, labels, and an adverse missing-return sensitivity check.
  6. Same locked penalties/smoothing for every ablation signal: the other combiner (whichever of
     pred_gate/pred_gate_notext/pred_ew/pred_ew_notext isn't the headline), pred_lgbm_all,
     pred_ridge, every pred_spec_*. The headline column is never backtested twice.
  7. evaluate.run_evaluation() for tables/charts; write the 3-file submission.
  8. Print the headline performance dict.
"""
import argparse
from pathlib import Path

import pandas as pd

from src import config, data, text, models, portfolio, evaluate


def run_signal(col, test_preds, panel, market, l2, tc):
    """Smooth preds column `col` into a signal and run the neutral optimizer backtest over it.
    Shared by the headline (config.HEADLINE_SIGNAL) and every ablation signal so they use
    identical smoothing and locked penalties."""
    sig = test_preds[['permno', 'eom']].copy()
    sig['signal'] = portfolio.smooth(test_preds, col)
    return portfolio.backtest(sig, panel, market, l2=l2, tc=tc)


def ablation_columns(headline_signal, specialist_cols):
    """Every ablation signal for step 6: the other combiner (whichever of pred_gate/
    pred_gate_notext/pred_ew/pred_ew_notext isn't `headline_signal`), plus the fixed and
    specialist ablations. Never includes `headline_signal` itself (A16: the headline column is
    never backtested twice)."""
    combiner_cols = ['pred_gate', 'pred_gate_notext', 'pred_ew', 'pred_ew_notext']
    combiner_ablations = [c for c in combiner_cols if c != headline_signal]
    fixed_ablations = combiner_ablations + ['pred_lgbm_all', 'pred_ridge']
    return fixed_ablations + sorted(specialist_cols)


def headline_r2_row(r2, headline_signal):
    """(model_col, oos_r2) for the headline's row in `r2` (models.r2_table output), or None if
    that column isn't present. pred_ew stays z-score-valued (A12): its return-unit twin
    pred_ew_ret carries the R2 instead. Any other headline (e.g. pred_gate, A16's pre-registered
    default) is itself return-unit and has its own row -- no substitution needed."""
    col = 'pred_ew_ret' if headline_signal == 'pred_ew' else headline_signal
    if col not in r2['model'].values:
        return None
    return col, r2.loc[r2['model'] == col, 'oos_r2'].iloc[0]


def main():
    parser = argparse.ArgumentParser(description='AlphaBERT end-to-end pipeline')
    parser.add_argument('--reuse-preds', action='store_true',
                         help='reuse cached outputs/cache/preds.parquet + gate_coefs.parquet instead of '
                              'recomputing models.run_all() (default: always recompute)')
    parser.add_argument('--dry', action='store_true',
                         help='load the cached panel, text features and market data, print shapes, then exit '
                              '(no model fitting, backtest, evaluation or submission)')
    args = parser.parse_args()

    # ---- 1. panel, market state, external market data -----------------------------------------
    print('=== step 1/8: build panel, market state, market data ===')
    panel = data.build_panel()
    state = data.market_state()
    market = data.load_market()

    # ---- 2. text features -----------------------------------------------------------------------
    print('=== step 2/8: text features ===')
    panel = text.add_text_features(panel)
    scores_path = text.scores_path_for(text.MAX_LENGTH)
    text_cols_used = [c for c in text.TEXT_FEATURES if c in panel.columns]
    tone_included = {'tone_mean', 'tone_min', 'fb_neg_max'}.issubset(set(text_cols_used))
    print(f'FinBERT cache ({scores_path.name}) present: {scores_path.exists()}; tone included: {tone_included}')
    print(f'text features in use: {text_cols_used}')

    if args.dry:
        print('=== --dry: skipping model fit / backtest / evaluate / submission ===')
        print(f'panel shape: {panel.shape}')
        print(f'market_state shape: {state.shape}')
        print(f'market shape: {market.shape}')
        return None

    # ---- 3. models: baselines, specialists, gate, per test year --------------------------------
    print('=== step 3/8: models.run_all (or reuse cached preds) ===')
    preds_path = config.CACHE_DIR / 'preds.parquet'
    if args.reuse_preds and preds_path.exists():
        preds = pd.read_parquet(preds_path)
        for c in ('eom', 'target_month'):
            if c in preds.columns:
                preds[c] = pd.to_datetime(preds[c]).astype('datetime64[ns]')
        missing = set(models.PRED_COLS) - set(preds.columns)
        models_path = Path(models.__file__)
        if missing:
            raise RuntimeError(
                f'--reuse-preds: cached {preds_path} is missing columns {sorted(missing)} that '
                f'models.PRED_COLS now expects -- this cache was built by an older src/models.py. '
                f'Re-run without --reuse-preds to rebuild it.')
        if preds_path.stat().st_mtime < models_path.stat().st_mtime:
            raise RuntimeError(
                f'--reuse-preds: cached {preds_path} is older than src/models.py -- the cache '
                f'predates the current model code. Re-run without --reuse-preds to rebuild it.')
        print(f'--reuse-preds set: loading cached {preds_path}')
    else:
        preds = models.run_all(panel, state)

    r2 = models.r2_table(preds)
    print('OOS R2 / IC table (test rows):')
    print(r2.to_string(index=False))
    headline_r2 = headline_r2_row(r2, config.HEADLINE_SIGNAL)
    if headline_r2 is not None:
        col, val = headline_r2
        print(f'headline OOS R2 ({col}, config.HEADLINE_SIGNAL={config.HEADLINE_SIGNAL}): {val}')
    r2.to_csv(config.TABLE_DIR / 'oos_r2.csv', index=False)

    # ---- 4. calibrate portfolio penalties on the SMOOTHED 2019-2020 validation pred_ew, lock ---
    print('=== step 4/8: calibrate portfolio penalties (2019-2020 validation, smoothed pred_ew) ===')
    valid_ew = preds.loc[preds['split'] == 'valid', ['permno', 'eom', 'pred_ew']].copy()
    valid_ew['signal'] = portfolio.smooth(valid_ew, 'pred_ew')
    valid_ew = valid_ew[['permno', 'eom', 'signal']]
    l2, tc, calib_table = portfolio.calibrate(valid_ew, panel, market)
    print(f'locked penalties: L2_PENALTY={l2}, TURNOVER_PENALTY={tc}')
    print(calib_table.to_string(index=False))
    calib_table.to_csv(config.TABLE_DIR / 'calibration.csv', index=False)

    # ---- 5. headline backtest: config.HEADLINE_SIGNAL, smoothed, formation months 2020-12..2026-07
    print(f'=== step 5/8: headline backtest ({config.HEADLINE_SIGNAL}) ===')
    test_preds = preds.loc[preds['split'] == 'test']
    holdings, returns = run_signal(config.HEADLINE_SIGNAL, test_preds, panel, market, l2, tc)
    label_panel, filing_labels = portfolio.load_label_sources()
    holdings = portfolio.attach_labels(holdings, label_panel, filing_labels)
    sensitivity = portfolio.missing_return_sensitivity(holdings, panel, returns)

    # ---- 6. ablations: same locked penalties, same smoothing, every alternative signal ---------
    # The other combiner is always reported as an ablation here, never as the headline: A16
    # reverted config.HEADLINE_SIGNAL to pred_gate (pre-registered ridge regime gate); pred_ew (the
    # post-hoc A12 headline) and its _notext variant are ablations now. If HEADLINE_SIGNAL were
    # switched back to 'pred_ew', pred_gate/pred_gate_notext would be the ablations instead -- the
    # headline column itself is never backtested twice.
    print('=== step 6/8: ablations ===')
    specialist_cols = [c for c in test_preds.columns if c.startswith('pred_spec_')]
    ablation_cols = ablation_columns(config.HEADLINE_SIGNAL, specialist_cols)
    ablations = {}
    ablation_failures = []
    for col in ablation_cols:
        try:
            _, ab_returns = run_signal(col, test_preds, panel, market, l2, tc)
        except RuntimeError as e:
            print(f'  ablation FAILED (infeasible under neutrality constraints): {col}: {e}')
            ablation_failures.append(dict(signal=col, error=str(e)))
            continue
        ablations[col] = ab_returns
        print(f'  ablation done: {col}')
    pd.DataFrame(ablation_failures, columns=['signal', 'error']).to_csv(
        config.TABLE_DIR / 'ablation_failures.csv', index=False)

    # ---- 7. evaluate + write submission ---------------------------------------------------------
    print('=== step 7/8: evaluate ===')
    gate_coefs = pd.read_parquet(config.CACHE_DIR / 'gate_coefs.parquet')
    headline = evaluate.run_evaluation(returns, holdings, panel, gate_coefs=gate_coefs, state=state,
                                        ablations=ablations, sensitivity_returns=sensitivity)

    print('=== step 8/8: write submission ===')
    portfolio.write_submission(holdings, returns)

    print('=== headline ===')
    print(headline)
    return headline


if __name__ == '__main__':
    main()
