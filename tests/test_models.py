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
    assert "pred_blend" in models.PRED_COLS and "pred_auto" in models.PRED_COLS
    assert {"pred_ew", "pred_ew_notext", "pred_blend", "pred_auto"}.isdisjoint(models.R2_COLS)

    expected_gate_cols = {"test_year", "specialist", "term", "coef",
                           "state_mean_mkt_vol12", "state_std_mkt_vol12",
                           "state_mean_disp", "state_std_disp"}
    assert set(gate.columns) == expected_gate_cols
    # 6 specialists (5 characteristic groups + text) each contribute a 'base' gate coefficient.
    assert set(gate.loc[gate["term"] == "base", "specialist"]) == set(models.CHAR_GROUPS) | {"text"}


def test_selected_features_and_headline_choice_reports(synthetic_run):
    """run_all writes TABLE_DIR/'selected_features.csv' (per-year selection report -- written
    whenever FEATURE_SELECTION or SELECTION_REPORT is True, which covers both current config
    defaults, A16) and TABLE_DIR/'headline_choice.csv' (per-year HEADLINE_CANDIDATES comparison,
    exactly one 'chosen' row per test_year)."""
    assert config.FEATURE_SELECTION or getattr(config, "SELECTION_REPORT", True)
    sel_path = config.TABLE_DIR / "selected_features.csv"
    headline_path = config.TABLE_DIR / "headline_choice.csv"
    assert sel_path.exists()
    assert headline_path.exists()

    sel = pd.read_csv(sel_path)
    assert {"feature", "group", "ic_t", "selected", "test_year"} <= set(sel.columns)
    assert set(sel["test_year"].unique()) == {2020}

    headline = pd.read_csv(headline_path)
    assert set(headline.columns) == {"test_year", "candidate", "valid_mean_ic", "chosen"}
    assert set(headline["candidate"].unique()) == set(config.HEADLINE_CANDIDATES)
    chosen_per_year = headline.groupby("test_year")["chosen"].sum()
    assert (chosen_per_year == 1).all()


def test_pred_blend_is_mean_of_zscored_ew_and_lgbm(synthetic_run):
    _, _, preds, _ = synthetic_run
    test_df = preds[preds["split"] == "test"]
    z_ew = models._zscore_by_eom(test_df["pred_ew"].values, test_df["eom"])
    z_lgbm = models._zscore_by_eom(test_df["pred_lgbm_all"].values, test_df["eom"])
    expected = (z_ew.values + z_lgbm.values) / 2.0
    assert np.allclose(test_df["pred_blend"].values, expected)


def test_headline_candidates_excludes_valid_fit_or_tuned_models():
    """config.HEADLINE_CANDIDATES must never include pred_gate/pred_gate_notext (fit on labelled
    valid rows) or pred_ols/pred_ridge/pred_lasso/pred_enet (alphas -- OLS trivially, but all four
    share the baseline family -- chosen/tuned on valid MSE): pred_auto's model choice must be
    genuinely out-of-sample on valid (A12), which only holds for candidates never fit/tuned there."""
    models._assert_headline_candidates_oos(config.HEADLINE_CANDIDATES)  # current config passes

    for bad in ("pred_gate", "pred_gate_notext", "pred_ols", "pred_ridge", "pred_lasso", "pred_enet"):
        with pytest.raises(AssertionError):
            models._assert_headline_candidates_oos(["pred_ew", bad])


def test_pred_auto_matches_best_valid_ic_candidate(synthetic_run):
    """pred_auto must equal whichever HEADLINE_CANDIDATES column headline_choice.csv marks
    'chosen' for that test_year -- both on valid and on test rows."""
    _, _, preds, _ = synthetic_run
    headline = pd.read_csv(config.TABLE_DIR / "headline_choice.csv")
    chosen = headline.loc[(headline["test_year"] == 2020) & headline["chosen"], "candidate"].item()
    for split in ("valid", "test"):
        sub = preds[preds["split"] == split]
        assert np.array_equal(sub["pred_auto"].values, sub[chosen].values, equal_nan=True)


def test_run_all_persists_valid_for_every_test_year(tmp_path):
    """Unlike the old first-year-only behaviour, run_all must now save split == 'valid' rows for
    EVERY test_year, not just the first."""
    orig_years, orig_end, orig_n_jobs = config.TEST_YEARS, config.TEST_END, config.N_JOBS
    config.TEST_YEARS = [2020, 2021]
    config.TEST_END = pd.Timestamp("2021-12-31")
    config.N_JOBS = 1
    try:
        panel = _make_panel(include_text=True, n_months=83, n_stocks=40)
        state = _make_state(panel["eom"])
        preds = models.run_all(panel, state, cache_dir=tmp_path / "run")

        valid_years = set(preds.loc[preds["split"] == "valid", "test_year"].unique())
        assert valid_years == {2020, 2021}
        for y in (2020, 2021):
            _, valid_mask, _ = models.splits(panel, y)
            got = (preds["split"] == "valid") & (preds["test_year"] == y)
            assert got.sum() == int(valid_mask.sum())
    finally:
        config.TEST_YEARS, config.TEST_END, config.N_JOBS = orig_years, orig_end, orig_n_jobs


def test_feature_selection_called_with_training_rows_only(tmp_path, monkeypatch):
    """selection.select_features must be called once per test_year with that year's TRAINING
    rows MINUS the inner-holdout months (the last INNER_HOLDOUT_MONTHS target months of train,
    later used by _fit_specialist to tune num_leaves/rounds) -- never valid/test rows, and never
    the inner-holdout months either, or the selection report's IC would be optimistic for the
    very rows tuning is scored on."""
    calls = []

    def fake_select_features(train_df, y, feature_cols, groups):
        calls.append((train_df.copy(), np.asarray(y).copy()))
        rows = []
        selected = []
        for g, cols in groups.items():
            for i, c in enumerate(cols):
                rows.append({"feature": c, "group": g, "ic_t": float(len(cols) - i)})
            selected.extend(cols[:3])
        report_df = pd.DataFrame(rows)
        return selected, report_df

    monkeypatch.setattr(models.selection, "select_features", fake_select_features)

    orig_years, orig_end, orig_n_jobs = config.TEST_YEARS, config.TEST_END, config.N_JOBS
    config.TEST_YEARS = [2020]
    config.TEST_END = pd.Timestamp("2020-12-31")
    config.N_JOBS = 1
    try:
        panel = _make_panel(seed=321, n_stocks=30, include_text=True)
        state = _make_state(panel["eom"])
        models.run_all(panel, state, cache_dir=tmp_path / "run")

        assert len(calls) == 1
        call_train_df, call_y = calls[0]
        train_mask, valid_mask, test_mask = models.splits(panel, 2020)
        expected_train = panel.loc[train_mask]
        train_months = np.sort(expected_train["target_month"].unique())
        n_ho = min(models.INNER_HOLDOUT_MONTHS, max(1, len(train_months) - 1))
        holdout_months = set(train_months[-n_ho:])
        expected_train = expected_train[~expected_train["target_month"].isin(holdout_months)]

        assert len(call_train_df) == len(expected_train)
        assert set(call_train_df.index) == set(expected_train.index)
        # never any valid/test row, and never an inner-holdout train row (spy on both)
        assert set(call_train_df.index).isdisjoint(set(panel.index[valid_mask]))
        assert set(call_train_df.index).isdisjoint(set(panel.index[test_mask]))
        assert set(call_train_df["target_month"].unique()).isdisjoint(holdout_months)
        expected_y = models.make_target(expected_train).values
        assert np.allclose(call_y, expected_y)
    finally:
        config.TEST_YEARS, config.TEST_END, config.N_JOBS = orig_years, orig_end, orig_n_jobs


def test_feature_selection_report_only_uses_full_features(tmp_path, monkeypatch):
    """A16: with FEATURE_SELECTION=False and SELECTION_REPORT=True, run_all must still call
    selection.select_features per test year and write TABLE_DIR/'selected_features.csv', but fit
    every model on the FULL feature set -- the report's returned subset must never be used to
    slice Xtr/Xva/Xte."""
    calls = []

    def fake_select_features(train_df, y, feature_cols, groups):
        calls.append(list(feature_cols))
        report_df = pd.DataFrame(
            [{"feature": c, "group": "value", "ic_t": 1.0} for c in feature_cols])
        return list(feature_cols[:3]), report_df  # deliberately tiny subset

    monkeypatch.setattr(models.selection, "select_features", fake_select_features)

    captured = {}
    orig_fit_baselines = models.fit_baselines

    def spy_fit_baselines(Xtr, ytr, Xva, yva):
        captured["n_cols"] = Xtr.shape[1]
        return orig_fit_baselines(Xtr, ytr, Xva, yva)

    monkeypatch.setattr(models, "fit_baselines", spy_fit_baselines)

    orig_years, orig_end, orig_n_jobs = config.TEST_YEARS, config.TEST_END, config.N_JOBS
    orig_fs = config.FEATURE_SELECTION
    had_sr = hasattr(config, "SELECTION_REPORT")
    orig_sr = getattr(config, "SELECTION_REPORT", True)
    config.TEST_YEARS = [2020]
    config.TEST_END = pd.Timestamp("2020-12-31")
    config.N_JOBS = 1
    config.FEATURE_SELECTION = False
    config.SELECTION_REPORT = True
    try:
        panel = _make_panel(seed=222, n_stocks=30, include_text=True)
        state = _make_state(panel["eom"])
        models.run_all(panel, state, cache_dir=tmp_path / "run")

        assert len(calls) == 1  # report still computed once for the one test_year
        from src.data import feature_columns
        n_full = len(feature_columns(panel))
        assert captured["n_cols"] == n_full  # models fit on the full feature set, not the tiny subset

        sel_path = config.TABLE_DIR / "selected_features.csv"
        assert sel_path.exists()
        sel = pd.read_csv(sel_path)
        assert set(sel["test_year"].unique()) == {2020}
        assert len(sel) == n_full  # report covers every candidate feature
    finally:
        config.TEST_YEARS, config.TEST_END, config.N_JOBS = orig_years, orig_end, orig_n_jobs
        config.FEATURE_SELECTION = orig_fs
        if had_sr:
            config.SELECTION_REPORT = orig_sr
        else:
            del config.SELECTION_REPORT


def test_ridge_alpha_search_subsampled_but_refit_uses_full_rows(monkeypatch):
    """A16: Ridge's alpha SEARCH is subsampled to MAX_ROWS_PENALIZED (like Lasso/ElasticNet)
    instead of running un-subsampled over the whole training window, but the final refit at the
    chosen alpha still uses the FULL training window."""
    n = models.MAX_ROWS_PENALIZED + 5_000
    rng = np.random.RandomState(0)
    Xtr = rng.normal(size=(n, 4)).astype(np.float32)
    true_w = np.array([1.0, -2.0, 0.5, 0.0])
    ytr = Xtr @ true_w + rng.normal(scale=0.1, size=n)
    Xva = rng.normal(size=(200, 4)).astype(np.float32)
    yva = Xva @ true_w + rng.normal(scale=0.1, size=200)

    fit_sizes = []
    orig_fit = models.Ridge.fit

    def spy_fit(self, X, y, *a, **k):
        fit_sizes.append(len(X))
        return orig_fit(self, X, y, *a, **k)

    monkeypatch.setattr(models.Ridge, "fit", spy_fit)

    out = models.fit_baselines(Xtr, ytr, Xva, yva)

    assert len(fit_sizes) >= 2  # at least one search fit + the final refit
    *search_sizes, final_size = fit_sizes
    assert all(s <= models.MAX_ROWS_PENALIZED for s in search_sizes), \
        "ridge alpha search must be subsampled to MAX_ROWS_PENALIZED like lasso/enet"
    assert final_size == n  # final refit uses the FULL training window
    assert out["ridge"].coef_.shape == (4,)


def test_ridge_alpha_grid_widened_to_include_1e8():
    """A16: chosen ridge alpha hit the old grid's 1e6 boundary every year, so RIDGE_ALPHAS is
    widened to np.logspace(-3, 8, 23)."""
    assert np.isclose(models.RIDGE_ALPHAS.max(), 1e8)
    assert np.isclose(models.RIDGE_ALPHAS.min(), 1e-3)
    assert len(models.RIDGE_ALPHAS) == 23


def test_pred_auto_ignores_test_labels(tmp_path):
    """Corrupting test-split stock_exret must not change pred_auto's choice or values --
    pred_auto is picked from labelled VALID rows only, walk-forward, and never touches test."""
    orig_years, orig_end, orig_n_jobs = config.TEST_YEARS, config.TEST_END, config.N_JOBS
    config.TEST_YEARS = [2020]
    config.TEST_END = pd.Timestamp("2020-12-31")
    config.N_JOBS = 1
    try:
        panel = _make_panel(seed=888, n_stocks=30, include_text=True)
        state = _make_state(panel["eom"])
        preds1 = models.run_all(panel, state, cache_dir=tmp_path / "run1")

        panel2 = panel.copy()
        _, _, test_mask = models.splits(panel2, 2020)
        rng = np.random.RandomState(0)
        panel2.loc[test_mask, "stock_exret"] = rng.normal(scale=50, size=int(test_mask.sum()))
        preds2 = models.run_all(panel2, state, cache_dir=tmp_path / "run2")

        t1 = preds1[preds1["split"] == "test"].sort_values(["permno", "eom"]).reset_index(drop=True)
        t2 = preds2[preds2["split"] == "test"].sort_values(["permno", "eom"]).reset_index(drop=True)
        assert np.array_equal(t1["pred_auto"].values, t2["pred_auto"].values, equal_nan=True)
        assert np.array_equal(t1["pred_blend"].values, t2["pred_blend"].values, equal_nan=True)
    finally:
        config.TEST_YEARS, config.TEST_END, config.N_JOBS = orig_years, orig_end, orig_n_jobs


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
    genuinely varied predictions -- check every test month has > 20 distinct pred_spec_quality
    values (90 stocks/month in the fixture, planted signal in gp_at/quality; FEATURE_SELECTION
    trims quality's ~50 mostly-noise characteristics down to a handful, which legitimately lowers
    the achievable distinct-value count vs. the full feature set, so the bar here is well above
    the single-digit count a truly degenerate 1-tree fit would produce, not a tight bound)."""
    _, _, preds, _ = synthetic_run
    test_df = preds[preds["split"] == "test"]
    nunique = test_df.groupby("eom")["pred_spec_quality"].nunique()
    assert (nunique > 20).all()


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
