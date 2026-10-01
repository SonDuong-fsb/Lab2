"""
Preprocessing and feature engineering.

  add_derived_features  works on the DataFrame and encodes domain knowledge.
  build_preprocessor    returns an unfitted sklearn transformer that is part of
                        the model Pipeline, so scaling and encoding are fitted on
                        training data only and travel with the model.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.config import (
    BILL_FEATURES,
    CATEGORICAL_FEATURES,
    PAY_AMT_FEATURES,
    PAY_FEATURES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RATIO_CAP = 5.0


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the six engineered features listed in config.DERIVED_FEATURES."""
    out = df.copy()

    avg_bill = out[BILL_FEATURES].mean(axis=1)
    avg_pay = out[PAY_AMT_FEATURES].mean(axis=1)
    limit = out["LIMIT_BAL"].replace(0, np.nan)
    last_bill = out["BILL_AMT1"].replace(0, np.nan)

    out["utilisation_ratio"] = (avg_bill / limit).clip(0, RATIO_CAP)
    out["payment_ratio"] = (out["PAY_AMT1"] / last_bill).clip(0, RATIO_CAP)
    out["max_delay"] = out[PAY_FEATURES].max(axis=1)
    out["n_months_delayed"] = (out[PAY_FEATURES] > 0).sum(axis=1)
    out["avg_bill_amt"] = avg_bill
    out["avg_pay_amt"] = avg_pay
    return out


def build_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    """Unfitted transformer: one-hot the categoricals, impute and scale the rest."""
    categorical = [c for c in CATEGORICAL_FEATURES if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_steps, numeric),
            ("categorical", categorical_encoder, categorical),
        ],
        remainder="drop",
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """The full feature step: derive, then drop nothing and let the model decide."""
    return add_derived_features(df)
