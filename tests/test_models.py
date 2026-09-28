import numpy as np
import pandas as pd
import pytest

from src import config, models

N_STOCKS = 90
N_MONTHS = 71  # eom 2015-01-31 .. 2020-11-30 -> target_month 2015-02-28..2020-12-31


def _calendar(n_months=N_MONTHS, start="2015-01-31"):
    return pd.date_range(start, periods=n_months, freq="ME")


def _make_panel(signal_col="gp_at", signal_strength=0.6, seed=123, include_text=False,
                 n_months=N_MONTHS, n_stocks=N_STOCKS):
    rng = np.random.RandomState(seed)
    eom = _calendar(n_months)
    permnos = np.arange(10000, 10000 + n_stocks)
    n = len(eom) * len(permnos)
    cols = {
        "permno": np.tile(permnos, len(eom)),
        "eom": np.repeat(eom, len(permnos)),
    }
    for c in config.load_char_list():
        cols[c] = rng.uniform(-1, 1, size=n)
    for c in ("gp_at", "ret_12_1"):
        cols[f"miss_{c}"] = rng.binomial(1, 0.05, size=n).astype("int8")

    if include_text:
        has_filing = (rng.uniform(size=n) < 0.3).astype(int)
        tone_mean = np.where(has_filing == 1, rng.normal(size=n), 0.0)
        cols["has_filing"] = has_filing
        cols["n_filings"] = has_filing
        for it in config.KEY_ITEMS:
            cols[f"item_{it.replace('.', '_')}"] = np.where(
                has_filing == 1, rng.poisson(1, size=n), 0)
        cols["tone_mean"] = tone_mean
        cols["tone_min"] = tone_mean
        cols["fb_neg_max"] = np.abs(tone_mean)

    df = pd.DataFrame(cols)
    df["target_month"] = df["eom"] + pd.offsets.MonthEnd(1)
    noise = rng.normal(scale=1.0, size=n)
    df["stock_exret"] = signal_strength * df[signal_col].values + 0.5 * noise
    return df


def _make_state(eom_values, seed=1):
    rng = np.random.RandomState(seed)
    eom_sorted = pd.DatetimeIndex(sorted(pd.unique(eom_values)), name="eom")
    return pd.DataFrame({
        "mkt_vol12": rng.uniform(0.05, 0.3, size=len(eom_sorted)),
        "disp": rng.uniform(0.01, 0.05, size=len(eom_sorted)),
    }, index=eom_sorted)


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    orig_years, orig_end = config.TEST_YEARS, config.TEST_END
    config.TEST_YEARS = [2020]
    config.TEST_END = pd.Timestamp("2020-12-31")
    cache_dir = tmp_path_factory.mktemp("models_cache")
    try:
        panel = _make_panel(include_text=True)
        state = _make_state(panel["eom"])
        preds = models.run_all(panel, state, cache_dir=cache_dir)
        gate = pd.read_parquet(cache_dir / "gate_coefs.parquet")
        yield panel, state, preds, gate
    finally:
        config.TEST_YEARS, config.TEST_END = orig_years, orig_end


# --------------------------------------------------------------------------
def test_char_groups_partition_all_147():
    names = set(config.load_char_list())
    grouped = []
    for g in models.CHAR_GROUPS:
        grouped.extend(models.CHAR_GROUPS[g])
    assert len(grouped) == 147
    assert len(set(grouped)) == 147  # no duplicates
    assert set(grouped) == names


def test_char_groups_has_five_specialists_with_investment_growth_moves():
    """A9: 5 characteristic groups; ebit_bev/sale_bev live in quality (not value)."""
    assert set(models.CHAR_GROUPS) == {
        "value", "momentum", "quality", "investment_growth", "risk_liquidity"}
    assert "ebit_bev" in models.CHAR_GROUPS["quality"]
    assert "sale_bev" in models.CHAR_GROUPS["quality"]
    assert "ebit_bev" not in models.CHAR_GROUPS["value"]
    assert "sale_bev" not in models.CHAR_GROUPS["value"]
    ig = set(models.CHAR_GROUPS["investment_growth"])
    assert {"at_gr1", "oaccruals_at", "mispricing_mgmt", "noa_at"} <= ig


def test_splits_schedule_matches_spec():
    eom = pd.date_range("2015-01-31", "2026-07-31", freq="ME")
    permnos = np.arange(5)
    df = pd.DataFrame({
        "permno": np.tile(permnos, len(eom)),
        "eom": np.repeat(eom, len(permnos)),
    })
    df["target_month"] = df["eom"] + pd.offsets.MonthEnd(1)
    df["stock_exret"] = np.random.RandomState(0).normal(size=len(df))

    test_month_union = set()
    for y in config.TEST_YEARS:
        train_mask, valid_mask, test_mask = models.splits(df, y)
        train_tm, valid_tm, test_tm = (df.loc[train_mask, "target_month"],
                                        df.loc[valid_mask, "target_month"],
                                        df.loc[test_mask, "target_month"])
        assert train_tm.max() < valid_tm.min() < test_tm.min()
        exp_start = pd.Timestamp(year=y, month=1, day=31)
        exp_end = min(pd.Timestamp(year=y, month=12, day=31), config.TEST_END)
        assert test_tm.min() == exp_start
        assert test_tm.max() == exp_end
        test_month_union |= set(test_tm.unique())

    expected_union = set(pd.date_range("2021-01-31", "2026-08-31", freq="ME"))
    assert test_month_union == expected_union


def test_train_excludes_null_label_valid_and_test_keep_it():
    """Train requires a label to fit; valid and test rows are predicted (and saved) even when
    stock_exret is null -- valid includes null-label future exits (A11), same as test."""
    eom = _calendar()
    permnos = np.arange(5)
    df = pd.DataFrame({
        "permno": np.tile(permnos, len(eom)),
        "eom": np.repeat(eom, len(permnos)),
    })
    df["target_month"] = df["eom"] + pd.offsets.MonthEnd(1)
    rng = np.random.RandomState(0)
    df["stock_exret"] = rng.normal(size=len(df))
    null_idx = rng.choice(len(df), size=len(df) // 5, replace=False)
    df.loc[df.index[null_idx], "stock_exret"] = np.nan

    train_mask, valid_mask, test_mask = models.splits(df, 2020)
    assert df.loc[train_mask, "stock_exret"].notna().all()
    assert df.loc[valid_mask, "stock_exret"].isna().any()
    assert df.loc[test_mask, "stock_exret"].isna().any()


def test_make_target_zero_mean_before_clip():
    rng = np.random.RandomState(0)
    tm = np.repeat(pd.date_range("2020-01-31", periods=6, freq="ME"), 50)
    df = pd.DataFrame({"target_month": tm, "stock_exret": rng.normal(size=len(tm))})
    df.loc[df.index[:3], "stock_exret"] = 100.0  # outliers so clipping actually engages

    demeaned = df["stock_exret"] - df.groupby("target_month")["stock_exret"].transform("mean")
    means_before_clip = demeaned.groupby(df["target_month"]).mean()
    assert np.allclose(means_before_clip.values, 0.0, atol=1e-8)

    target = models.make_target(df)
    assert not np.allclose(target.values, demeaned.values)  # clip actually changed something


def test_make_target_ignores_null_label_rows():
    """A11: null-label rows must not distort the demeaning/clipping stats used for labelled rows
    in the same target_month."""
    rng = np.random.RandomState(0)
    tm = np.repeat(pd.date_range("2020-01-31", periods=6, freq="ME"), 50)
    df = pd.DataFrame({"target_month": tm, "stock_exret": rng.normal(size=len(tm))})
    target_labelled_only = models.make_target(df)

    df2 = df.copy()
    extra = pd.DataFrame({"target_month": pd.date_range("2020-01-31", periods=6, freq="ME"),
                           "stock_exret": np.nan})
    df2 = pd.concat([df2, extra], ignore_index=True)
    target_with_nulls = models.make_target(df2)

    assert np.allclose(target_with_nulls.iloc[:len(df)].values, target_labelled_only.values)
    assert target_with_nulls.iloc[len(df):].isna().all()


def test_oos_r2_formula():
    y = np.array([1.0, -2.0, 3.0, 0.5])
    yhat = np.array([0.8, -1.5, 2.0, 0.1])
    expected = 1 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)
    assert np.isclose(models.oos_r2(y, yhat), expected)


def test_zscore_by_eom_zero_for_nan_rows():
    """_zscore_by_eom already implements the filer-only z-score used for the text specialist:
    NaN raw predictions (non-filers) are ignored by the group mean/std, then fillna(0) sets their
    own z-score to exactly 0 (this is why the old _text_z helper was redundant and was removed)."""
    eom = pd.Series(pd.to_datetime(["2020-01-31"] * 6 + ["2020-02-29"] * 6))
    raw = np.array([1.0, 2.0, 3.0, np.nan, np.nan, np.nan,
                     -1.0, 0.0, 1.0, np.nan, np.nan, np.nan])
    z = models._zscore_by_eom(raw, eom)
    nonfiler = np.isnan(raw)
    assert (z[nonfiler] == 0.0).all()
    assert abs(z[~nonfiler][:3].mean()) < 1e-8  # filers z-scored within month -> ~0 mean


def test_specialist_ignores_valid_labels():
    """Corrupting valid-window stock_exret must not change the specialist's test-set predictions
    (specialists are tuned/fit on train + an inner train holdout only, per A2 -- see the bigger
    run_all-level version of this check, test_run_all_ignores_valid_labels, below)."""
    orig_n_jobs = config.N_JOBS
    config.N_JOBS = 1
    try:
        panel = _make_panel(seed=555, n_stocks=60)
        train_mask, valid_mask, _ = models.splits(panel, 2020)
        train = panel.loc[train_mask].copy()
        cols = models.CHAR_GROUPS["quality"]
        ytr = models.make_target(train).values
        m1 = models._fit_specialist(train, ytr, cols)

        panel2 = panel.copy()
        garbage = np.random.RandomState(0).normal(scale=50, size=int(valid_mask.sum()))
        panel2.loc[valid_mask, "stock_exret"] = garbage
        train2_mask, valid2_mask, test2_mask = models.splits(panel2, 2020)
        assert valid2_mask.equals(valid_mask)
        assert train2_mask.equals(train_mask)

        train2 = panel2.loc[train2_mask].copy()
        ytr2 = models.make_target(train2).values
        m2 = models._fit_specialist(train2, ytr2, cols)

        Xte = panel.loc[models.splits(panel, 2020)[2], cols].astype(np.float32).values
        pred1, pred2 = m1.predict(Xte), m2.predict(Xte)
        assert np.allclose(pred1, pred2, atol=1e-6)
    finally:
        config.N_JOBS = orig_n_jobs


@pytest.mark.slow
def test_shuffled_label_ic_small():
    """A single shuffled-label fit can land a modest (~0.02-0.07) spurious IC purely from
    gradient boosting overfitting its inner-holdout noise onto one of the ~50 quality features by
    chance (this does not shrink with a bigger n_stocks -- it is a per-fit event, not sampling
    noise). What DOES shrink reliably is averaging over several INDEPENDENT shuffles/fits: each
    rep's spurious feature/direction is drawn independently, so the pooled mean IC over enough
    reps is a stable, tight-tolerance null check. n_stocks=400 keeps each rep's own sampling noise
    small too."""
    n_reps, n_stocks = 16, 300
    all_ic = []
    cols = models.CHAR_GROUPS["quality"]
    for rep in range(n_reps):
        panel = _make_panel(seed=9000 + rep, n_stocks=n_stocks)
        train_mask, _, test_mask = models.splits(panel, 2020)
        train, test = panel.loc[train_mask].copy(), panel.loc[test_mask].copy()
        rng = np.random.RandomState(rep)
        train["stock_exret"] = (
            train.groupby("eom")["stock_exret"].transform(lambda s: rng.permutation(s.values))
        )
        ytr = models.make_target(train).values
        m = models._fit_specialist(train, ytr, cols)
        pred_test = m.predict(test[cols].astype(np.float32).values)
        all_ic.append(models.monthly_ic(test.assign(pred=pred_test), "pred").dropna())

    pooled_mean_ic = pd.concat(all_ic).mean()
    assert abs(pooled_mean_ic) < 0.02


def test_run_all_ignores_valid_labels(tmp_path):
    """Corrupting valid-window labels must not change any specialist, lgbm_all, or OLS test-set
    prediction -- none of them are ever fit or tuned on valid labels (A2: specialists/lgbm_all are
    tuned on an inner TRAIN holdout only; OLS has no alpha, so it never touches valid at all).
    pred_ew/pred_ew_notext are unaffected too, since they are plain means of test-set specialist
    z-scores. Ridge/Lasso/ElasticNet and the gate are EXPECTED to change: A2 explicitly allows
    linear-baseline alpha selection to use the official valid window (they are template baselines,
    not gate inputs), and the gate is fit on the labelled valid subset by design."""
    orig_years, orig_end = config.TEST_YEARS, config.TEST_END
    orig_n_jobs = config.N_JOBS
    config.TEST_YEARS = [2020]
    config.TEST_END = pd.Timestamp("2020-12-31")
    config.N_JOBS = 1
    try:
        panel = _make_panel(seed=777, n_stocks=30, include_text=True)
        state = _make_state(panel["eom"])
        preds1 = models.run_all(panel, state, cache_dir=tmp_path / "run1")

        panel2 = panel.copy()
        _, valid_mask, _ = models.splits(panel2, 2020)
        rng = np.random.RandomState(0)
        panel2.loc[valid_mask, "stock_exret"] = rng.normal(scale=50, size=int(valid_mask.sum()))
        preds2 = models.run_all(panel2, state, cache_dir=tmp_path / "run2")

        t1 = preds1[preds1["split"] == "test"].sort_values(["permno", "eom"]).reset_index(drop=True)
        t2 = preds2[preds2["split"] == "test"].sort_values(["permno", "eom"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(t1[["permno", "eom"]], t2[["permno", "eom"]])

        spec_cols = [f"pred_spec_{g}" for g in models.CHAR_GROUPS] + ["pred_spec_text"]
        unaffected = spec_cols + ["pred_lgbm_all", "pred_ols", "pred_ew", "pred_ew_notext"]
        for c in unaffected:
            assert np.array_equal(t1[c].values, t2[c].values, equal_nan=True), \
                f"{c} changed when only valid labels were corrupted"

        may_change = ["pred_ridge", "pred_lasso", "pred_enet", "pred_gate", "pred_gate_notext"]
        assert any(not np.array_equal(t1[c].values, t2[c].values, equal_nan=True) for c in may_change), \
            "expected at least one valid-dependent column (ridge/lasso/enet/gate) to change"
    finally:
        config.TEST_YEARS, config.TEST_END = orig_years, orig_end
        config.N_JOBS = orig_n_jobs


def test_run_all_valid_includes_null_label_rows(tmp_path):
    """A11: the saved 'valid' split (calibration window) includes rows with a null stock_exret
    (future exits), and predictions are made for them too, not just for labelled rows."""
    orig_years, orig_end = config.TEST_YEARS, config.TEST_END
    orig_n_jobs = config.N_JOBS
    config.TEST_YEARS = [2020]
    config.TEST_END = pd.Timestamp("2020-12-31")
    config.N_JOBS = 1
    try:
        panel = _make_panel(seed=42, n_stocks=30, include_text=True)
        _, valid_mask, _ = models.splits(panel, 2020)
        valid_idx = panel.index[valid_mask]
        rng = np.random.RandomState(0)
        null_idx = rng.choice(valid_idx, size=len(valid_idx) // 10, replace=False)
        panel.loc[null_idx, "stock_exret"] = np.nan

        state = _make_state(panel["eom"])
        preds = models.run_all(panel, state, cache_dir=tmp_path / "run")

        valid_out = preds[preds["split"] == "valid"]
        assert len(valid_out) == int(valid_mask.sum())
        assert valid_out["stock_exret"].isna().sum() == len(null_idx)
        null_out = valid_out[valid_out["stock_exret"].isna()]
        assert null_out["pred_gate"].notna().all()
        assert null_out["pred_spec_quality"].notna().all()
    finally:
        config.TEST_YEARS, config.TEST_END = orig_years, orig_end
        config.N_JOBS = orig_n_jobs


# --------------------------------------------------- run_all (shared fixture)
def test_run_all_schema(synthetic_run):
    _, _, preds, gate = synthetic_run
    expected_cols = ({"permno", "eom", "target_month", "stock_exret", "test_year", "split"}
                      | set(models.PRED_COLS))
    assert set(preds.columns) == expected_cols
    assert set(preds["split"].unique()) <= {"valid", "test"}
    assert set(preds.loc[preds["split"] == "valid", "test_year"].unique()) == {2020}
    assert set(preds["test_year"].unique()) == {2020}

    expected_gate_cols = {"test_year", "specialist", "term", "coef",
                           "state_mean_mkt_vol12", "state_std_mkt_vol12",
                           "state_mean_disp", "state_std_disp"}
    assert set(gate.columns) == expected_gate_cols
    # 6 specialists (5 characteristic groups + text) each contribute a 'base' gate coefficient.
    assert set(gate.loc[gate["term"] == "base", "specialist"]) == set(models.CHAR_GROUPS) | {"text"}


def test_planted_signal_specialist_and_gate(synthetic_run):
    _, _, preds, gate = synthetic_run
    test_df = preds[preds["split"] == "test"]

    ic_quality = models.monthly_ic(test_df, "pred_spec_quality").mean()
    assert ic_quality > 0.05

    ic_others = [models.monthly_ic(test_df, f"pred_spec_{g}").mean()
                 for g in ("value", "momentum", "investment_growth", "risk_liquidity")]
    assert ic_quality > max(ic_others)

    gate_quality_base = gate[(gate["specialist"] == "quality") & (gate["term"] == "base")]
    assert len(gate_quality_base) == 1
    assert (gate_quality_base["coef"] > 0).all()


def test_specialist_not_degenerate(synthetic_run):
    """A13: MSE-based early stopping was found to select degenerate low-round/low-leaf models
    (e.g. a single tree with 7 distinct predictions). The IC-selected specialist must produce
    genuinely varied predictions -- check every test month has > 50 distinct pred_spec_quality
    values (150 stocks/month in the fixture, planted signal in gp_at/quality)."""
    _, _, preds, _ = synthetic_run
    test_df = preds[preds["split"] == "test"]
    nunique = test_df.groupby("eom")["pred_spec_quality"].nunique()
    assert (nunique > 50).all()


def test_r2_table_nan_for_ew_columns(synthetic_run):
    _, _, preds, _ = synthetic_run
    r2 = models.r2_table(preds)
    r2 = r2.set_index("model")
    for c in ("pred_ew", "pred_ew_notext"):
        assert np.isnan(r2.loc[c, "oos_r2"])
        assert np.isnan(r2.loc[c, "oos_r2_demeaned"])
        assert not np.isnan(r2.loc[c, "mean_ic"])  # IC is still reported for ew/ew_notext
    for c in ("pred_gate", "pred_lgbm_all", "pred_spec_quality"):
        assert not np.isnan(r2.loc[c, "oos_r2"])
    assert not np.isnan(r2.loc["pred_ew_ret", "oos_r2"])  # return-unit blend: R2 IS meaningful


def test_pred_ew_ret_is_raw_specialist_mean(synthetic_run):
    """pred_ew_ret = row-mean of the six RAW specialist forecasts (return units, not z-scores),
    with the text specialist's non-filer NaN treated as 0 (headline OOS R2 disclosure, brief
    p.20 / SPEC section 5)."""
    _, _, preds, _ = synthetic_run
    spec_cols = [f"pred_spec_{g}" for g in models.CHAR_GROUPS] + ["pred_spec_text"]
    raw = preds[spec_cols].copy()
    raw["pred_spec_text"] = raw["pred_spec_text"].fillna(0.0)
    expected = raw.mean(axis=1)
    assert np.allclose(preds["pred_ew_ret"].values, expected.values)


@pytest.mark.slow
def test_real_panel_ignores_test_labels(tmp_path):
    """Slow: on a subset of permnos from the real panel (one test year), corrupting test-split
    stock_exret must not change any prediction column -- test labels are never used to fit or
    predict, only for evaluation (r2_table), which run_all never touches."""
    from src import data, text as text_mod

    orig_years, orig_end = config.TEST_YEARS, config.TEST_END
    orig_n_jobs = config.N_JOBS
    config.TEST_YEARS = [2021]
    config.N_JOBS = 4
    try:
        panel = text_mod.add_text_features(data.build_panel())
        state = data.market_state()
        rng = np.random.RandomState(0)
        uniq_permnos = panel["permno"].unique()
        keep = rng.choice(uniq_permnos, size=min(400, len(uniq_permnos)), replace=False)
        panel = panel[panel["permno"].isin(keep)].reset_index(drop=True)

        preds1 = models.run_all(panel, state, cache_dir=tmp_path / "real1")

        panel2 = panel.copy()
        _, _, test_mask = models.splits(panel2, 2021)
        rng2 = np.random.RandomState(1)
        panel2.loc[test_mask, "stock_exret"] = rng2.normal(scale=50, size=int(test_mask.sum()))
        preds2 = models.run_all(panel2, state, cache_dir=tmp_path / "real2")

        t1 = preds1[preds1["split"] == "test"].sort_values(["permno", "eom"]).reset_index(drop=True)
        t2 = preds2[preds2["split"] == "test"].sort_values(["permno", "eom"]).reset_index(drop=True)
        for c in models.PRED_COLS:
            assert np.array_equal(t1[c].values, t2[c].values, equal_nan=True), c
    finally:
        config.TEST_YEARS, config.TEST_END = orig_years, orig_end
        config.N_JOBS = orig_n_jobs
