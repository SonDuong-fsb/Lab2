"""
Data ingestion stage.

Loads the raw dataset and produces a reproducible train/test split. The split
is stratified on the target and seeded
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import DATA_PATH, RANDOM_STATE, TARGET, TEST_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw dataset from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 'python scripts/make_dataset.py' first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %s rows x %s columns from %s", len(df), df.shape[1], path)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info("Split: train=%s test=%s", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Summary of the dataset, logged to MLflow and passed between Airflow tasks."""
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "positive_rate": round(float(df[TARGET].mean()), 4),
        "n_missing": int(df.isna().sum().sum()),
    }


def load_and_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """Convenience wrapper: load, summarise and split in one call."""
    df = load_raw()
    stats = dataset_stats(df)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test, stats
