"""
Configuration for the credit default ML pipeline.

"""

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

for _d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(os.getenv("DATA_PATH", str(DATA_DIR / "credit_default.csv")))

# =============================================================================
# MLflow
# =============================================================================
# Default is a local file store, so the pipeline runs with no server at all —
# that is what CI uses. docker-compose overrides it with the tracking server.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{BASE_DIR / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-default-risk")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "credit-default-classifier")

# MLflow deprecated model registry stages in 2.9. Aliases replace them: a moving
# pointer to one version, which you can repoint atomically without touching the
# versions themselves.
CHAMPION_ALIAS = "champion"      # what production serves
CHALLENGER_ALIAS = "challenger"  # the candidate under evaluation

# =============================================================================
# Data
# =============================================================================
TARGET = "default_payment_next_month"
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]

RAW_FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_FEATURES + BILL_FEATURES + PAY_AMT_FEATURES
)

# Features created in preprocessing.add_derived_features
DERIVED_FEATURES = [
    "utilisation_ratio",
    "payment_ratio",
    "max_delay",
    "n_months_delayed",
    "avg_bill_amt",
    "avg_pay_amt",
]

TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 501))

# Attribute used for the fairness slice in evaluation. Session 6 explains why
# this belongs in the pipeline and not in a notebook someone runs once.
SENSITIVE_ATTRIBUTE = "SEX"

# =============================================================================
# Data validation thresholds
# =============================================================================
MAX_MISSING_FRACTION = 0.02      # per column
MIN_ROWS = 5_000
MIN_POSITIVE_RATE = 0.05         # a degenerate target means something upstream broke
MAX_POSITIVE_RATE = 0.60

# =============================================================================
# Models
# =============================================================================
DEFAULT_MODEL_TYPE = "hgb"

MODEL_CONFIGS = {
    "logreg": {"C": 1.0, "max_iter": 1000},
    "rf": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20},
    "hgb": {"max_iter": 300, "learning_rate": 0.06, "max_depth": 6, "l2_regularization": 1.0},
}

# The sweep run by experiments/run_experiments.py
EXPERIMENT_GRID = [
    {"model_type": "logreg", "C": 0.1, "max_iter": 1000},
    {"model_type": "logreg", "C": 1.0, "max_iter": 1000},
    {"model_type": "rf", "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20},
    {"model_type": "rf", "n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20},
    {"model_type": "hgb", "max_iter": 200, "learning_rate": 0.10, "max_depth": 4, "l2_regularization": 1.0},
    {"model_type": "hgb", "max_iter": 300, "learning_rate": 0.06, "max_depth": 6, "l2_regularization": 1.0},
    {"model_type": "hgb", "max_iter": 500, "learning_rate": 0.03, "max_depth": 8, "l2_regularization": 2.0},
]

# =============================================================================
# Promotion gate
# =============================================================================
# A model is only promoted to champion if it clears these. Writing the gate down
# is what turns "we looked at the numbers and it seemed fine" into something a
# scheduled pipeline can decide on its own at 3 a.m.
PRIMARY_METRIC = "roc_auc"
MIN_ROC_AUC = float(os.getenv("MIN_ROC_AUC", 0.70))
MIN_PR_AUC = float(os.getenv("MIN_PR_AUC", 0.45))
MAX_FAIRNESS_GAP = float(os.getenv("MAX_FAIRNESS_GAP", 0.10))

# Decision thresholds, kept identical to Lab 1 so the two labs agree.
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", 0.30))

# =============================================================================
# Airflow
# =============================================================================
AIRFLOW_DAG_ID = "credit_default_training"
AIRFLOW_SCHEDULE = os.getenv("AIRFLOW_SCHEDULE", "@weekly")
