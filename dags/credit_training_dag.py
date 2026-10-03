"""
Airflow DAG: scheduled retraining for the credit default model.

    ingest -> validate -> train -> evaluate -> decide -> [promote | skip] -> cleanup
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The image installs the project at /opt/airflow/project; this makes the local
# checkout work too, so the DAG can be parsed outside the container.
sys.path.insert(0, os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

RUN_DIR = Path(os.getenv("PIPELINE_RUN_DIR", "/opt/airflow/artifacts"))

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# =============================================================================
# Tasks
# =============================================================================
def ingest(**context):
    """Load the raw dataset and stash the split on the shared volume."""
    import joblib

    from pipeline.data_ingestion import dataset_stats, load_raw, split_data

    run_dir = RUN_DIR / context["run_id"].replace(":", "_").replace("+", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    stats = dataset_stats(df)
    X_train, X_test, y_train, y_test = split_data(df)

    joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
                run_dir / "split.joblib")
    df.to_parquet(run_dir / "raw.parquet", index=False)

    ti = context["ti"]
    ti.xcom_push(key="run_dir", value=str(run_dir))
    ti.xcom_push(key="data_stats", value=stats)
    return f"{stats['n_rows']} rows, positive rate {stats['positive_rate']}"


def validate(**context):
    """Quality gate on the data. Raising here stops the DAG before training."""
    import pandas as pd

    from pipeline.validation import validate_dataset

    ti = context["ti"]
    run_dir = Path(ti.xcom_pull(task_ids="ingest", key="run_dir"))

    df = pd.read_parquet(run_dir / "raw.parquet")
    report = validate_dataset(df, raise_on_error=True)

    (run_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    ti.xcom_push(key="validation_report", value=report)
    return f"validated {report['n_rows']} rows"


def train(**context):
    """Fit the pipeline inside an MLflow run."""
    import joblib

    from pipeline.training import setup_mlflow, train_model

    ti = context["ti"]
    run_dir = Path(ti.xcom_pull(task_ids="ingest", key="run_dir"))
    split = joblib.load(run_dir / "split.joblib")

    setup_mlflow()
    model, mlflow_run_id = train_model(
        split["X_train"],
        split["y_train"],
        model_type=os.getenv("MODEL_TYPE", "hgb"),
        run_name=f"airflow-{context['ds']}",
        data_stats=ti.xcom_pull(task_ids="ingest", key="data_stats"),
        validation_report=ti.xcom_pull(task_ids="validate", key="validation_report"),
    )

    joblib.dump(model, run_dir / "model.joblib")
    ti.xcom_push(key="mlflow_run_id", value=mlflow_run_id)
    return mlflow_run_id


def evaluate(**context):
    """Score the held-out set and log every metric to the run."""
    import joblib

    from pipeline.evaluation import evaluate_model
    from pipeline.training import setup_mlflow

    ti = context["ti"]
    run_dir = Path(ti.xcom_pull(task_ids="ingest", key="run_dir"))
    split = joblib.load(run_dir / "split.joblib")
    model = joblib.load(run_dir / "model.joblib")

    setup_mlflow()
    result = evaluate_model(
        model,
        split["X_test"],
        split["y_test"],
        run_id=ti.xcom_pull(task_ids="train", key="mlflow_run_id"),
    )

    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
    ti.xcom_push(key="metrics", value=metrics)
    return f"roc_auc={metrics['roc_auc']:.4f} fairness_gap={metrics['fairness_gap']:.4f}"


def decide(**context):
    """Branch: does this model clear the promotion gate?"""
    from pipeline.registry import passes_quality_gate

    ti = context["ti"]
    gate = passes_quality_gate(ti.xcom_pull(task_ids="evaluate", key="metrics"))
    ti.xcom_push(key="quality_gate", value=gate)
    return "promote_model" if gate["passed"] else "skip_promotion"


def promote(**context):
    """Register the run and give it the alias it earned."""
    from pipeline.registry import promote_model
    from pipeline.training import setup_mlflow

    ti = context["ti"]
    setup_mlflow()
    result = promote_model(
        ti.xcom_pull(task_ids="train", key="mlflow_run_id"),
        ti.xcom_pull(task_ids="evaluate", key="metrics"),
    )

    ti.xcom_push(key="promotion", value=result)
    return f"{result['model_name']} v{result['version']} -> {result['outcome']}"


def cleanup(**context):
    """Remove the run directory. Runs whether or not the model was promoted."""
    run_dir = context["ti"].xcom_pull(key="run_dir")
    if run_dir and Path(run_dir).exists():
        shutil.rmtree(run_dir, ignore_errors=True)
        return f"removed {run_dir}"
    return "nothing to clean"


# =============================================================================
# DAG
# =============================================================================
with DAG(
    dag_id="credit_default_training",
    default_args=default_args,
    description="Scheduled retraining and gated promotion for the credit default model",
    schedule=os.getenv("AIRFLOW_SCHEDULE", "@weekly"),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "credit-risk"],
) as dag:

    t_ingest = PythonOperator(task_id="ingest", python_callable=ingest)
    t_validate = PythonOperator(task_id="validate", python_callable=validate)
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate)
    t_decide = BranchPythonOperator(task_id="decide", python_callable=decide)
    t_promote = PythonOperator(task_id="promote_model", python_callable=promote)
    t_skip = PythonOperator(task_id="skip_promotion", python_callable=promote)

    # none_failed_min_one_success: cleanup must run down whichever branch was
    # taken, but must not run if an upstream task actually failed.
    t_cleanup = PythonOperator(
        task_id="cleanup",
        python_callable=cleanup,
        trigger_rule="none_failed_min_one_success",
    )

    t_ingest >> t_validate >> t_train >> t_evaluate >> t_decide
    t_decide >> [t_promote, t_skip] >> t_cleanup
