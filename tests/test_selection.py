"""Synthetic-panel tests for src.selection.select_features: 3 planted signal characteristics
among 30 noise characteristics, ~40 months x 300 stocks, deterministic given config.SEED."""
import time

import numpy as np
import pandas as pd

from src import config
from src.selection import select_features

N_MONTHS = 40
N_STOCKS = 300
N_NOISE_PER_GROUP = 10

SIGNAL_CHARS = ["sig1", "sig2", "sig3"]
GROUPS = {
    "a": ["sig1"] + [f"noise_a{i:02d}" for i in range(N_NOISE_PER_GROUP)],
    "b": ["sig2"] + [f"noise_b{i:02d}" for i in range(N_NOISE_PER_GROUP)],
    "c": ["sig3"] + [f"noise_c{i:02d}" for i in range(N_NOISE_PER_GROUP)],
}
NOISE_CHARS = [c for cs in GROUPS.values() for c in cs if c not in SIGNAL_CHARS]
CHAR_COLS = SIGNAL_CHARS + NOISE_CHARS  # 3 signal + 30 noise = 33


def _make_panel():
    rng = np.random.RandomState(config.SEED)
    months = pd.date_range("2015-01-31", periods=N_MONTHS, freq="ME")
    n = N_MONTHS * N_STOCKS

    df = pd.DataFrame(rng.normal(size=(n, len(CHAR_COLS))), columns=CHAR_COLS)
    df["target_month"] = np.repeat(months, N_STOCKS)
    df["permno"] = np.tile(np.arange(N_STOCKS), N_MONTHS)

    eps = rng.normal(scale=2.0, size=n)
    y = 0.6 * df["sig1"].to_numpy() - 0.5 * df["sig2"].to_numpy() + 0.4 * df["sig3"].to_numpy() + eps
    y = pd.Series(y, index=df.index)

    # miss_ flags: one on a signal char, one on a noise char, unrelated to y (random 0/1)
    df["miss_sig1"] = (rng.uniform(size=n) < 0.1).astype("int8")
    df[f"miss_{NOISE_CHARS[0]}"] = (rng.uniform(size=n) < 0.1).astype("int8")

    feature_cols = CHAR_COLS + ["miss_sig1", f"miss_{NOISE_CHARS[0]}"]
    return df, y, feature_cols


def test_select_features_recovers_planted_signal_is_fast_and_deterministic():
    train_df, y, feature_cols = _make_panel()

    t0 = time.time()
    selected_cols, report_df = select_features(train_df, y, feature_cols, GROUPS)
    elapsed = time.time() - t0
    assert elapsed < 30, f"select_features took {elapsed:.1f}s, expected < 30s"

    # report_df shape/columns
    assert list(report_df.columns) == [
        "feature", "group", "ic_mean", "ic_t", "p_value", "bh_pass", "lasso_freq", "selected",
    ]
    assert set(report_df["feature"]) == set(feature_cols)

    # planted signals recovered
    for c in SIGNAL_CHARS:
        assert c in selected_cols, f"planted signal {c} was not selected"

    # most noise dropped
    noise_selected = [c for c in NOISE_CHARS if c in selected_cols]
    assert len(noise_selected) <= 0.3 * len(NOISE_CHARS), (
        f"too many noise features selected: {noise_selected}")

    # min-per-group respected
    for g, chars in GROUPS.items():
        n_sel = sum(1 for c in chars if c in selected_cols)
        assert n_sel >= config.SELECTION_MIN_PER_GROUP, f"group {g} under min: {n_sel}"

    # miss_ flags follow their underlying char
    rep = report_df.set_index("feature")["selected"]
    assert rep["miss_sig1"] == rep["sig1"]
    noise0 = NOISE_CHARS[0]
    assert rep[f"miss_{noise0}"] == rep[noise0]

    # deterministic given the same rows/seed, and uses only the rows passed in
    selected_cols2, report_df2 = select_features(train_df, y, feature_cols, GROUPS)
    assert selected_cols2 == selected_cols
    pd.testing.assert_frame_equal(report_df, report_df2)

    # a disjoint subset of rows (different months) gives a self-contained, independent result
    half_months = train_df["target_month"].drop_duplicates().iloc[: N_MONTHS // 2]
    sub_mask = train_df["target_month"].isin(half_months)
    sub_selected, sub_report = select_features(
        train_df.loc[sub_mask], y.loc[sub_mask], feature_cols, GROUPS)
    assert set(sub_report["feature"]) == set(feature_cols)
    for c in SIGNAL_CHARS:
        assert c in sub_selected, f"planted signal {c} not recovered on the row subset"
