"""Cheap, pure-function tests for MAIN.py's headline/ablation selection logic (headline is
config.HEADLINE_SIGNAL; the ablation list is the fixed set of alternative combiners plus every
specialist, minus whichever one is the headline). These do not run the pipeline -- see
tests/test_integrity.py::test_mini_end_to_end_pipeline for that."""
import pandas as pd

from MAIN import ablation_columns, headline_r2_row


FIXED_ABLATIONS = ('pred_ew', 'pred_lgbm_all', 'pred_blend', 'pred_gate', 'pred_gate_notext',
                    'pred_ew_notext', 'pred_ridge')


def test_ablation_columns_headline_pred_auto():
    """Pre-registered default headline (pred_auto, not itself a fixed-ablation column) ->
    every fixed ablation is present, plus the specialists."""
    cols = ablation_columns('pred_auto', ['pred_spec_value', 'pred_spec_momentum'])
    assert 'pred_auto' not in cols
    for c in FIXED_ABLATIONS + ('pred_spec_value', 'pred_spec_momentum'):
        assert c in cols
    assert len(cols) == len(set(cols))


def test_ablation_columns_excludes_headline_when_headline_is_a_fixed_ablation():
    """When the headline signal IS one of the fixed ablation columns (e.g. pred_blend, an
    alternative headline choice), it must never appear twice -- excluded from the ablation list,
    every other fixed column still present."""
    cols = ablation_columns('pred_blend', ['pred_spec_value'])
    assert 'pred_blend' not in cols
    for c in ('pred_ew', 'pred_lgbm_all', 'pred_gate', 'pred_gate_notext', 'pred_ew_notext',
              'pred_ridge', 'pred_spec_value'):
        assert c in cols


def test_ablation_columns_headline_pred_gate():
    cols = ablation_columns('pred_gate', ['pred_spec_value'])
    assert 'pred_gate' not in cols
    for c in ('pred_ew', 'pred_lgbm_all', 'pred_blend', 'pred_gate_notext', 'pred_ew_notext',
              'pred_ridge', 'pred_spec_value'):
        assert c in cols


def test_ablation_columns_specialists_sorted():
    cols = ablation_columns('pred_auto', ['pred_spec_risk_liquidity', 'pred_spec_momentum'])
    specialists = [c for c in cols if c.startswith('pred_spec_')]
    assert specialists == sorted(specialists)


def test_ablation_columns_no_duplicates_across_headlines():
    """Every fixed-ablation value used as `headline_signal` yields a list with no duplicates and
    without itself."""
    for headline in FIXED_ABLATIONS:
        cols = ablation_columns(headline, [])
        assert headline not in cols
        assert len(cols) == len(set(cols))
        assert len(cols) == len(FIXED_ABLATIONS) - 1


def test_headline_r2_row_pred_ew_uses_ret_twin():
    """headline=pred_ew: the return-unit twin pred_ew_ret carries the R2; pred_ew itself
    (z-score-valued) is not looked up even though it's also a row in r2."""
    r2 = pd.DataFrame({'model': ['pred_ew', 'pred_ew_ret', 'pred_auto'], 'oos_r2': [float('nan'), 0.01, 0.02]})
    result = headline_r2_row(r2, 'pred_ew')
    assert result == ('pred_ew_ret', 0.01)


def test_headline_r2_row_pred_auto_is_its_own_row():
    """headline=pred_auto (pre-registered default): pred_auto is return-unit and already has an
    R2 in the table -- no _ret substitution."""
    r2 = pd.DataFrame({'model': ['pred_ew', 'pred_ew_ret', 'pred_auto'], 'oos_r2': [float('nan'), 0.01, 0.02]})
    result = headline_r2_row(r2, 'pred_auto')
    assert result == ('pred_auto', 0.02)


def test_headline_r2_row_missing_column_returns_none():
    r2 = pd.DataFrame({'model': ['pred_ew'], 'oos_r2': [float('nan')]})
    assert headline_r2_row(r2, 'pred_ew') is None
