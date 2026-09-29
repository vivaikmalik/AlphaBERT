"""A16 (docs/SPEC.md section 10): config-level switches for the two amendments (A12 headline,
A15 beta model) that were reverted because they were decided after test-period numbers had been
seen, plus the N_JOBS env override. See src/config.py and docs/research_log.md."""
import importlib

from src import config


def test_pre_registered_defaults():
    """Headline is the walk-forward auto-selected signal. Beta model defaults to 'fusion'
    (chosen on a validation-only horse race, 2019-2020 validation + 2016-18 pseudo-history, no
    test data seen -- see docs/research_log.md); 'blume' remains selectable and is the
    pre-registered path."""
    assert config.HEADLINE_SIGNAL == 'pred_auto'
    assert config.BETA_MODEL == 'fusion'


def test_net_mode_defaults():
    """NET_MODE defaults to 'dollar': the same validation-only horse race found a flexible net
    exposure did not improve realized book beta and drifted the book net long with the signal,
    so beta neutrality is instead reached via a nonzero BETA_TARGET (see
    test_beta_target_default) rather than a flexible net. 'beta' mode remains selectable and
    keeps NET_CAP/NET_PENALTY sane for when it's used."""
    assert config.NET_MODE == 'dollar'
    assert 0 < config.NET_CAP <= 0.50
    assert config.NET_PENALTY > 0
    assert not hasattr(config, 'BETA_OVERLAY')


def test_n_cand_tracks_beta_model():
    """N_CAND is 350 whenever BETA_MODEL != 'blume' -- the default 'fusion' beta model included --
    and 250 only under the pre-registered 'blume' model."""
    assert config.BETA_MODEL == 'fusion'
    assert config.N_CAND == 350


def test_beta_target_default():
    """BETA_TARGET (new): a small positive ex-ante beta target, chosen on 2016-2018
    pseudo-history and confirmed on 2019-2020 validation (no test data), corrects the
    systematic negative realized book beta seen at a target of 0. BETA_TOL is tightened
    alongside it."""
    assert config.BETA_TARGET == 0.075
    assert config.BETA_TOL == 0.005


def test_feature_selection_defaults():
    """Validation (2019-2020) showed per-window factor selection LOWERS IC, so it no longer
    restricts the model's features by default; SELECTION_REPORT keeps the per-window
    "which factors matter" diagnostic table computed for reporting only."""
    assert config.FEATURE_SELECTION is False
    assert config.SELECTION_REPORT is True


def test_n_jobs_env_override(monkeypatch):
    """N_JOBS reads ALPHABERT_NJOBS at import time (falls back to 6 when unset) so users on other
    machines never need to edit config.py directly."""
    monkeypatch.delenv('ALPHABERT_NJOBS', raising=False)
    importlib.reload(config)
    try:
        assert config.N_JOBS == 6

        monkeypatch.setenv('ALPHABERT_NJOBS', '3')
        importlib.reload(config)
        assert config.N_JOBS == 3
    finally:
        monkeypatch.delenv('ALPHABERT_NJOBS', raising=False)
        importlib.reload(config)
        assert config.N_JOBS == 6
