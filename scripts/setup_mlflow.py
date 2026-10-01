"""
Check that MLflow is reachable and the experiment exists.

Run this before the first pipeline run to separate "MLflow is not up" from
"my pipeline code is wrong" — two failures that look identical in a stack trace.

Usage:
    python scripts/setup_mlflow.py
"""

import sys

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)


def main() -> None:
    print(f"Tracking URI : {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        client = MlflowClient()
        client.search_experiments(max_results=1)
    except Exception as exc:  # noqa: BLE001
        print(f"\nCannot reach the tracking server: {exc}")
        print("\nIf you expected a server, start it with:  docker compose up -d mlflow")
        print("To work without one, unset MLFLOW_TRACKING_URI and the pipeline")
        print("will use a local file store at ./mlruns")
        sys.exit(1)

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"Created experiment '{MLFLOW_EXPERIMENT_NAME}' (id {experiment_id})")
    else:
        print(f"Experiment   : {MLFLOW_EXPERIMENT_NAME} (id {experiment.experiment_id})")

    models = [m.name for m in client.search_registered_models()]
    print(f"Registered   : {models or 'none yet'}")
    print(f"Target name  : {REGISTERED_MODEL_NAME}")
    print("\nMLflow is ready.")


if __name__ == "__main__":
    main()
