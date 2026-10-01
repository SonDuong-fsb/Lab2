"""
Training stage, with MLflow tracking.
"""

import hashlib
import logging
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipeline.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_CONFIGS,
    RANDOM_STATE,
    TEST_SIZE,
)
from pipeline.preprocessing import build_preprocessor, prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "logreg": LogisticRegression,
    "rf": RandomForestClassifier,
    "hgb": HistGradientBoostingClassifier,
}


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> str:
    """Point MLflow at the tracking store and make sure the experiment exists."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking at %s, experiment '%s'", tracking_uri, experiment_name)
    return tracking_uri


def dataset_fingerprint() -> str:
    if not DATA_PATH.exists():
        return "unavailable"
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:16]


def build_model(model_type: str, **params: Any) -> Any:
    """Instantiate an estimator by name."""
    cls = MODEL_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available: {sorted(MODEL_CLASSES)}"
        )
    merged = {**MODEL_CONFIGS.get(model_type, {}), **params}
    if "random_state" not in merged:
        merged["random_state"] = RANDOM_STATE
    if model_type == "rf":
        merged.setdefault("n_jobs", -1)
    return cls(**merged)


def build_pipeline(model_type: str, feature_columns: list, **params: Any) -> Pipeline:
    """Preprocessor + estimator as a single fitted-together object."""
    return Pipeline([
        ("preprocess", build_preprocessor(feature_columns)),
        ("classifier", build_model(model_type, **params)),
    ])


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "hgb",
    run_name: Optional[str] = None,
    data_stats: Optional[Dict[str, Any]] = None,
    validation_report: Optional[Dict[str, Any]] = None,
    **params: Any,
) -> Tuple[Pipeline, str]:
    """Fit a pipeline inside an MLflow run and return it with the run id."""
    X_train = prepare_features(X_train)
    feature_columns = list(X_train.columns)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "model_type": model_type,
            "n_features": len(feature_columns),
            "n_train_rows": len(X_train),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "data_sha256": dataset_fingerprint(),
            **params,
        })

        if data_stats:
            mlflow.log_params({f"data_{key}": value for key, value in data_stats.items()})

        if validation_report:
            mlflow.log_dict(validation_report, "validation_report.json")
            mlflow.set_tag("validation_passed", str(validation_report.get("passed", False)).lower())

        mlflow.log_dict({"features": feature_columns}, "feature_columns.json")

        pipeline = build_pipeline(model_type, feature_columns, **params)
        pipeline.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_train.head(3).astype("float64"),
        )
        mlflow.set_tag("stage", "trained")

    logger.info("Trained %s in run %s", model_type, run_id)
    return pipeline, run_id
