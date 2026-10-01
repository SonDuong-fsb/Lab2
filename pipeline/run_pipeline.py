"""
End-to-end pipeline runner.

    ingest -> validate -> train -> evaluate -> promote

Usage:
    python -m pipeline.run_pipeline
    python -m pipeline.run_pipeline --model-type rf --no-register
    python -m pipeline.run_pipeline --model-type hgb --max-iter 500 --run-name nightly
"""

import argparse
import json
import logging
from typing import Any, Dict

from pipeline.config import (
    ARTIFACTS_DIR,
    DEFAULT_MODEL_TYPE,
    MLFLOW_EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)
from pipeline.data_ingestion import load_and_split, load_raw
from pipeline.evaluation import evaluate_model
from pipeline.registry import promote_model
from pipeline.training import setup_mlflow, train_model
from pipeline.validation import validate_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def run(
    model_type: str = DEFAULT_MODEL_TYPE,
    run_name: str = None,
    register: bool = True,
    **model_params: Any,
) -> Dict[str, Any]:
    """Run every stage once and return a summary."""
    setup_mlflow()

    logger.info("[1/5] Ingest")
    raw = load_raw()

    logger.info("[2/5] Validate")
    report = validate_dataset(raw)

    logger.info("[3/5] Split and train")
    X_train, X_test, y_train, y_test, stats = load_and_split()
    model, run_id = train_model(
        X_train, y_train,
        model_type=model_type,
        run_name=run_name,
        data_stats=stats,
        validation_report=report,
        **model_params,
    )

    logger.info("[4/5] Evaluate")
    metrics = evaluate_model(model, X_test, y_test, run_id=run_id)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "model_type": model_type,
        "experiment": MLFLOW_EXPERIMENT_NAME,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "group_metrics": metrics.get("group_metrics", {}),
    }

    if register:
        logger.info("[5/5] Promote")
        summary["promotion"] = promote_model(run_id, metrics, REGISTERED_MODEL_NAME)
    else:
        logger.info("[5/5] Promote — skipped (--no-register)")
        summary["promotion"] = None

    out = ARTIFACTS_DIR / "last_run.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE,
                        choices=["logreg", "rf", "hgb"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-register", action="store_true",
                        help="train and evaluate but do not touch the registry")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    params = {
        k: v for k, v in {
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        }.items() if v is not None
    }

    summary = run(
        model_type=args.model_type,
        run_name=args.run_name,
        register=not args.no_register,
        **params,
    )

    print("\n" + "=" * 62)
    print(f"run_id   {summary['run_id']}")
    print(f"model    {summary['model_type']}")
    m = summary["metrics"]
    print(f"roc_auc  {m['roc_auc']:.4f}    pr_auc {m['pr_auc']:.4f}")
    print(f"recall   {m['recall']:.4f}    fairness gap {m['fairness_gap']:.4f}")
    if summary["promotion"]:
        p = summary["promotion"]
        print(f"registry {p['model_name']} v{p['version']} -> {p['outcome']}")
        if not p["quality_gate"]["passed"]:
            print(f"         gate failed: {p['quality_gate']['failed_checks']}")
    print("=" * 62)


if __name__ == "__main__":
    main()
