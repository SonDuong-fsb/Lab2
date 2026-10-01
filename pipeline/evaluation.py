"""
Evaluation stage.

  Aggregate metrics   ROC AUC, PR AUC, precision/recall at the decision threshold.
  Sliced metrics      the same numbers computed separately per group.
"""

import logging
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipeline.config import REVIEW_THRESHOLD, SENSITIVE_ATTRIBUTE
from pipeline.preprocessing import prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_GROUP_ROWS = 50


def compute_metrics(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float = REVIEW_THRESHOLD
) -> Dict[str, float]:
    """Aggregate metrics at a fixed decision threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "threshold": float(threshold),
    }


def compute_group_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    groups: pd.Series,
    threshold: float = REVIEW_THRESHOLD,
) -> Dict[str, Dict[str, float]]:
    """The same metrics, one set per group value."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    groups = pd.Series(np.asarray(groups))

    result: Dict[str, Dict[str, float]] = {}
    for value in sorted(groups.dropna().unique()):
        mask = (groups == value).to_numpy()
        if mask.sum() < MIN_GROUP_ROWS or len(np.unique(y_true[mask])) < 2:
            logger.warning("Skipping group %s=%s: too small or single-class", SENSITIVE_ATTRIBUTE, value)
            continue
        metrics = compute_metrics(y_true[mask], y_proba[mask], threshold)
        result[str(value)] = {
            **metrics,
            "n": int(mask.sum()),
            "selection_rate": float((y_proba[mask] >= threshold).mean()),
        }
    return result


def fairness_gap(group_metrics: Dict[str, Dict[str, float]], key: str = "selection_rate") -> float:
    """Largest difference in `key` between any two groups."""
    values = [metrics[key] for metrics in group_metrics.values() if key in metrics]
    if len(values) < 2:
        return 0.0
    return float(max(values) - min(values))


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_id: Optional[str] = None,
    threshold: float = REVIEW_THRESHOLD,
) -> Dict[str, Any]:
    """Score the test set, compute everything, and log it to the run."""
    sensitive = X_test[SENSITIVE_ATTRIBUTE].copy() if SENSITIVE_ATTRIBUTE in X_test else None
    X_prepared = prepare_features(X_test)

    y_proba = model.predict_proba(X_prepared)[:, 1]
    metrics = compute_metrics(y_test, y_proba, threshold)

    group_metrics: Dict[str, Dict[str, float]] = {}
    gap = 0.0
    if sensitive is not None:
        group_metrics = compute_group_metrics(y_test, y_proba, sensitive, threshold)
        gap = fairness_gap(group_metrics)

    result = {**metrics, "fairness_gap": gap, "group_metrics": group_metrics}

    logger.info(
        "ROC AUC %.4f | PR AUC %.4f | recall %.3f | fairness gap %.3f",
        metrics["roc_auc"], metrics["pr_auc"], metrics["recall"], gap,
    )

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_metric("fairness_gap", gap)
            for value, gm in group_metrics.items():
                for k, v in gm.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"group_{SENSITIVE_ATTRIBUTE}_{value}_{k}", v)
            mlflow.log_dict(result, "evaluation.json")
            mlflow.set_tag("stage", "evaluated")

    return result
