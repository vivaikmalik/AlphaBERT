"""Cheap, pure-function tests for MAIN.py's headline/ablation selection logic (A16: headline is
config.HEADLINE_SIGNAL, and the ablation list always includes the other combiner). These do not
run the pipeline -- see tests/test_integrity.py::test_mini_end_to_end_pipeline for that."""
import pandas as pd

from MAIN import ablation_columns, headline_r2_row


def test_ablation_columns_headline_pred_gate():
    """Pre-registered default (A16): headline pred_gate -> ablations include pred_ew,
    pred_ew_notext and pred_gate_notext, but never pred_gate itself."""
    cols = ablation_columns('pred_gate', ['pred_spec_value', 'pred_spec_momentum'])
    assert 'pred_gate' not in cols
    for c in ('pred_ew', 'pred_ew_notext', 'pred_gate_notext', 'pred_lgbm_all', 'pred_ridge',
              'pred_spec_value', 'pred_spec_momentum'):
        assert c in cols
    assert len(cols) == len(set(cols))


def test_ablation_columns_headline_pred_ew():
    """A12 alternative: headline pred_ew -> ablations include pred_gate and pred_gate_notext
    (and pred_ew_notext), but never pred_ew itself."""
    cols = ablation_columns('pred_ew', ['pred_spec_value'])
    assert 'pred_ew' not in cols
    for c in ('pred_gate', 'pred_gate_notext', 'pred_ew_notext', 'pred_lgbm_all', 'pred_ridge',
              'pred_spec_value'):
        assert c in cols


def test_ablation_columns_specialists_sorted():
    cols = ablation_columns('pred_gate', ['pred_spec_risk_liquidity', 'pred_spec_momentum'])
    specialists = [c for c in cols if c.startswith('pred_spec_')]
    assert specialists == sorted(specialists)


def test_headline_r2_row_pred_ew_uses_ret_twin():
    """headline=pred_ew: the return-unit twin pred_ew_ret carries the R2 (A12); pred_ew itself
    (z-score-valued) is not looked up even though it's also a row in r2."""
    r2 = pd.DataFrame({'model': ['pred_ew', 'pred_ew_ret', 'pred_gate'], 'oos_r2': [float('nan'), 0.01, 0.02]})
    result = headline_r2_row(r2, 'pred_ew')
    assert result == ('pred_ew_ret', 0.01)


def test_headline_r2_row_pred_gate_is_its_own_row():
    """headline=pred_gate (A16 pre-registered default): pred_gate is return-unit and already has
    an R2 in the table -- no _ret substitution."""
    r2 = pd.DataFrame({'model': ['pred_ew', 'pred_ew_ret', 'pred_gate'], 'oos_r2': [float('nan'), 0.01, 0.02]})
    result = headline_r2_row(r2, 'pred_gate')
    assert result == ('pred_gate', 0.02)


def test_headline_r2_row_missing_column_returns_none():
    r2 = pd.DataFrame({'model': ['pred_ew'], 'oos_r2': [float('nan')]})
    assert headline_r2_row(r2, 'pred_ew') is None
