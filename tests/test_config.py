"""A16 (docs/SPEC.md section 10): config-level switches for the two amendments (A12 headline,
A15 beta model) that were reverted because they were decided after test-period numbers had been
seen, plus the N_JOBS env override. See src/config.py and docs/research_log.md."""
import importlib

from src import config


def test_pre_registered_defaults():
    """A16 default: the pre-registered headline (pred_gate) and beta model (blume)."""
    assert config.HEADLINE_SIGNAL == 'pred_gate'
    assert config.BETA_MODEL == 'blume'


def test_n_cand_tracks_beta_model():
    """N_CAND is 250 under the pre-registered ('blume') beta model -- A15 raised it to 350 only
    together with its own (now ablation-only) beta model."""
    assert config.BETA_MODEL == 'blume'
    assert config.N_CAND == 250


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
