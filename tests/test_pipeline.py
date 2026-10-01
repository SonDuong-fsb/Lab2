"""
Tests for the credit default pipeline.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=pipeline --cov-report=term-missing

These run against a temporary MLflow file store, so they need no server and
leave nothing behind.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from pipeline.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    DERIVED_FEATURES,
    PAY_FEATURES,
    RAW_FEATURES,
    TARGET,
)
from pipeline.data_ingestion import dataset_stats, load_raw, split_data
from pipeline.evaluation import compute_group_metrics, compute_metrics, fairness_gap
from pipeline.preprocessing import add_derived_features, build_preprocessor
from pipeline.registry import (
    find_best_run,
    get_model_version_by_alias,
    passes_quality_gate,
    promote_model,
)
from pipeline.training import build_model, build_pipeline
from pipeline.validation import DataValidationError, validate_dataset


# =============================================================================
@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """The real dataset, loaded once for the whole module."""
    return load_raw()


@pytest.fixture(scope="module")
def small(raw) -> pd.DataFrame:
    """A stratified 3,000-row sample — enough to fit, fast enough for CI."""
    return raw.groupby(TARGET, group_keys=False).apply(
        lambda g: g.sample(min(len(g), 1500), random_state=0)
    ).reset_index(drop=True)


# =============================================================================
class TestDataIngestion:
    """Loading and splitting."""

    def test_dataset_has_expected_shape(self, raw):
        assert len(raw) == 30_000
        assert TARGET in raw.columns
        assert all(c in raw.columns for c in RAW_FEATURES)

    def test_split_is_stratified(self, raw):
        X_train, X_test, y_train, y_test = split_data(raw)
        assert abs(y_train.mean() - y_test.mean()) < 0.01

    def test_split_is_reproducible(self, raw):
        a = split_data(raw)[0].index.tolist()
        b = split_data(raw)[0].index.tolist()
        assert a == b, "same seed must give the same split, or metrics are not comparable"

    def test_no_leakage_between_train_and_test(self, raw):
        X_train, X_test, _, _ = split_data(raw)
        assert not set(X_train.index) & set(X_test.index)

    def test_stats_report_positive_rate(self, raw):
        s = dataset_stats(raw)
        assert 0.05 < s["positive_rate"] < 0.60
        assert s["n_rows"] == len(raw)


# =============================================================================
class TestValidation:
    """The quality gate in front of training."""

    @pytest.fixture
    def frame(self, raw):
        return raw.copy()

    def test_clean_data_passes(self, raw):
        report = validate_dataset(raw)
        assert report["passed"] is True
        assert report["n_errors"] == 0

    def test_missing_column_is_caught(self, frame):
        with pytest.raises(DataValidationError, match="missing columns"):
            validate_dataset(frame.drop(columns=["LIMIT_BAL"]))

    def test_non_numeric_column_is_caught(self, frame):
        frame["AGE"] = frame["AGE"].astype(str)
        with pytest.raises(DataValidationError, match="non-numeric"):
            validate_dataset(frame)

    def test_too_few_rows_is_caught(self, frame):
        with pytest.raises(DataValidationError, match="too few rows"):
            validate_dataset(frame.head(100))

    def test_missing_values_are_caught(self, frame):
        frame.loc[frame.index[:1000], "BILL_AMT1"] = np.nan
        with pytest.raises(DataValidationError, match="BILL_AMT1"):
            validate_dataset(frame)

    def test_out_of_domain_category_is_caught(self, frame):
        frame.loc[frame.index[:5], "SEX"] = 7
        with pytest.raises(DataValidationError, match="SEX"):
            validate_dataset(frame)

    def test_impossible_age_is_caught(self, frame):
        frame.loc[frame.index[:5], "AGE"] = 400
        with pytest.raises(DataValidationError, match="AGE"):
            validate_dataset(frame)

    def test_negative_payment_is_caught(self, frame):
        frame.loc[frame.index[:5], "PAY_AMT1"] = -100
        with pytest.raises(DataValidationError, match="PAY_AMT1"):
            validate_dataset(frame)

    def test_degenerate_target_is_caught(self, frame):
        frame[TARGET] = 0
        with pytest.raises(DataValidationError, match="positive rate"):
            validate_dataset(frame)

    def test_all_levels_are_reported_together(self, frame):
        frame[TARGET] = 0
        frame.loc[frame.index[:5], "SEX"] = 7
        report = validate_dataset(frame, raise_on_error=False)
        assert report["statistical_errors"]
        assert report["semantic_errors"]
        assert report["n_errors"] == len(report["statistical_errors"]) + len(report["semantic_errors"])

    def test_report_is_returned_when_not_raising(self, frame):
        frame.loc[frame.index[:5], "AGE"] = 400
        report = validate_dataset(frame, raise_on_error=False)
        assert report["passed"] is False
        assert any("AGE" in error for error in report["semantic_errors"])


# =============================================================================
class TestPreprocessing:
    """Feature engineering and the transformer."""

    def test_derived_features_are_added(self, small):
        out = add_derived_features(small)
        assert set(DERIVED_FEATURES) <= set(out.columns)

    def test_original_frame_is_not_mutated(self, small):
        before = list(small.columns)
        add_derived_features(small)
        assert list(small.columns) == before

    def test_utilisation_ratio_is_sensible(self, small):
        ratio = add_derived_features(small)["utilisation_ratio"].dropna()
        assert ratio.between(0, 5).all()

    def test_zero_limit_gives_nan_not_inf(self, small):
        frame = small.head(10).copy()
        frame["LIMIT_BAL"] = 0
        assert add_derived_features(frame)["utilisation_ratio"].isna().all()

    def test_max_delay_matches_pay_columns(self, small):
        out = add_derived_features(small)
        assert (out["max_delay"] == small[PAY_FEATURES].max(axis=1)).all()

    def test_months_delayed_counts_positive_statuses(self, small):
        out = add_derived_features(small)
        assert (out["n_months_delayed"] == (small[PAY_FEATURES] > 0).sum(axis=1)).all()

    def test_preprocessor_one_hots_categoricals(self, small):
        X = add_derived_features(small.drop(columns=[TARGET]))
        transformed = build_preprocessor(list(X.columns)).fit_transform(X)
        assert transformed.shape[0] == len(X)
        assert transformed.shape[1] > X.shape[1]

    def test_preprocessor_is_returned_unfitted(self, small):
        X = add_derived_features(small.drop(columns=[TARGET]))
        assert not hasattr(build_preprocessor(list(X.columns)), "transformers_")

    def test_preprocessor_handles_unseen_category(self, small):
        X = add_derived_features(small.drop(columns=[TARGET]))
        preprocessor = build_preprocessor(list(X.columns)).fit(X)
        unseen = X.head(5).copy()
        unseen["EDUCATION"] = 99
        assert preprocessor.transform(unseen).shape[0] == 5


# =============================================================================
class TestTraining:
    """Model construction and fitting."""

    @pytest.mark.parametrize("model_type", ["logreg", "rf", "hgb"])
    def test_every_model_type_builds(self, model_type):
        assert build_model(model_type) is not None

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError):
            build_model("does-not-exist")

    def test_params_override_defaults(self):
        assert build_model("hgb", max_iter=7).max_iter == 7

    def test_pipeline_fits_and_predicts_probabilities(self, small):
        X = add_derived_features(small.drop(columns=[TARGET]))
        pipe = build_pipeline("logreg", list(X.columns))
        pipe.fit(X, small[TARGET])
        proba = pipe.predict_proba(X)[:, 1]
        assert proba.shape == (len(X),)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_preprocessing_travels_with_the_model(self):
        pipe = build_pipeline("hgb", RAW_FEATURES)
        assert [name for name, _ in pipe.steps] == ["preprocess", "classifier"]


# =============================================================================
class TestEvaluation:
    """Metrics, slices and the fairness gap."""

    @pytest.fixture(scope="class")
    def scored(self, small):
        X = add_derived_features(small.drop(columns=[TARGET]))
        y = small[TARGET]
        pipe = build_pipeline("hgb", list(X.columns), max_iter=40)
        pipe.fit(X, y)
        return y, pipe.predict_proba(X)[:, 1], small["SEX"]

    def test_metrics_are_in_range(self, scored):
        y, proba, _ = scored
        metrics = compute_metrics(y, proba)
        for key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
            assert 0.0 <= metrics[key] <= 1.0, key

    def test_confusion_counts_sum_to_n(self, scored):
        y, proba, _ = scored
        m = compute_metrics(y, proba)
        total = m["true_positives"] + m["false_positives"] + m["false_negatives"] + m["true_negatives"]
        assert total == len(y)

    def test_lower_threshold_never_lowers_recall(self, scored):
        y, proba, _ = scored
        assert compute_metrics(y, proba, 0.2)["recall"] >= compute_metrics(y, proba, 0.5)["recall"]

    def test_group_metrics_cover_every_group(self, scored):
        y, proba, sex = scored
        groups = compute_group_metrics(y, proba, sex)
        assert set(groups) == {"1", "2"}
        assert sum(g["n"] for g in groups.values()) == len(y)

    def test_small_group_is_skipped(self, scored):
        y, proba, sex = scored
        sex = sex.copy()
        sex.iloc[:10] = 9
        assert "9" not in compute_group_metrics(y, proba, sex)

    def test_fairness_gap_is_non_negative(self, scored):
        y, proba, sex = scored
        assert fairness_gap(compute_group_metrics(y, proba, sex)) >= 0.0

    def test_fairness_gap_is_the_selection_rate_range(self):
        groups = {
            "1": {"selection_rate": 0.10},
            "2": {"selection_rate": 0.35},
            "3": {"selection_rate": 0.20},
        }
        assert fairness_gap(groups) == pytest.approx(0.25)

    def test_fairness_gap_is_zero_for_one_group(self):
        assert fairness_gap({"1": {"selection_rate": 0.4}}) == 0.0


# =============================================================================
class TestQualityGate:
    """The promotion rule, tested without touching a registry."""

    def test_good_model_passes(self):
        gate = passes_quality_gate({"roc_auc": 0.75, "pr_auc": 0.55, "fairness_gap": 0.03})
        assert gate["passed"] is True
        assert gate["failed_checks"] == []

    def test_weak_model_is_rejected(self):
        gate = passes_quality_gate({"roc_auc": 0.60, "pr_auc": 0.30, "fairness_gap": 0.03})
        assert gate["passed"] is False
        assert {"roc_auc", "pr_auc"} <= set(gate["failed_checks"])

    def test_accurate_but_unfair_model_is_rejected(self):
        gate = passes_quality_gate({"roc_auc": 0.90, "pr_auc": 0.80, "fairness_gap": 0.40})
        assert gate["passed"] is False
        assert gate["failed_checks"] == ["fairness_gap"]

    def test_missing_metric_fails_closed(self):
        gate = passes_quality_gate({})
        assert gate["passed"] is False
        assert set(gate["failed_checks"]) == {"roc_auc", "pr_auc", "fairness_gap"}

    def test_missing_fairness_gap_alone_fails(self):
        assert passes_quality_gate({"roc_auc": 0.9, "pr_auc": 0.9})["passed"] is False


# =============================================================================
class TestPromotion:
    """Registry outcomes, against a throwaway file store."""

    GOOD = {"roc_auc": 0.80, "pr_auc": 0.60, "fairness_gap": 0.05}

    @pytest.fixture(autouse=True)
    def tracking_store(self, tmp_path):
        mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
        mlflow.set_experiment("promotion-tests")
        yield
        mlflow.set_tracking_uri(None)

    @staticmethod
    def logged_run(metrics):
        model = LogisticRegression().fit([[0.0], [1.0], [0.0], [1.0]], [0, 1, 0, 1])
        with mlflow.start_run() as run:
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model", pip_requirements=["scikit-learn"])
        return run.info.run_id

    def test_first_passing_model_becomes_champion(self):
        result = promote_model(self.logged_run(self.GOOD), self.GOOD, "gate-model")
        assert result["outcome"] == "champion"
        assert str(get_model_version_by_alias("gate-model", CHAMPION_ALIAS)["version"]) == result["version"]

    def test_failing_model_is_registered_but_not_aliased(self):
        unfair = {**self.GOOD, "fairness_gap": 0.40}
        result = promote_model(self.logged_run(unfair), unfair, "gate-model")
        assert result["outcome"] == "rejected"
        assert get_model_version_by_alias("gate-model", CHAMPION_ALIAS) is None
        assert get_model_version_by_alias("gate-model", CHALLENGER_ALIAS) is None
        version = mlflow.MlflowClient().get_model_version("gate-model", result["version"])
        assert version.tags["quality_gate"] == "failed"

    def test_marginal_improvement_becomes_challenger(self):
        promote_model(self.logged_run(self.GOOD), self.GOOD, "gate-model")
        close = {**self.GOOD, "roc_auc": 0.801}
        result = promote_model(self.logged_run(close), close, "gate-model")
        assert result["outcome"] == "challenger"
        assert str(get_model_version_by_alias("gate-model", CHAMPION_ALIAS)["version"]) == "1"

    def test_clear_improvement_replaces_champion(self):
        promote_model(self.logged_run(self.GOOD), self.GOOD, "gate-model")
        better = {**self.GOOD, "roc_auc": 0.85}
        result = promote_model(self.logged_run(better), better, "gate-model")
        assert result["outcome"] == "champion"
        assert str(get_model_version_by_alias("gate-model", CHAMPION_ALIAS)["version"]) == "2"

    def test_best_run_follows_the_metric(self):
        self.logged_run({"roc_auc": 0.70})
        best = self.logged_run({"roc_auc": 0.90})
        assert find_best_run("promotion-tests")["run_id"] == best

    def test_unknown_experiment_raises(self):
        with pytest.raises(ValueError):
            find_best_run("no-such-experiment")
