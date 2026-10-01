"""
Data validation stage — the quality gate in front of training.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from pipeline.config import (
    MAX_MISSING_FRACTION,
    MAX_POSITIVE_RATE,
    MIN_POSITIVE_RATE,
    MIN_ROWS,
    RAW_FEATURES,
    TARGET,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the dataset fails a check that must not be ignored."""


DOMAINS: Dict[str, Any] = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
}
RANGES: Dict[str, tuple] = {
    "LIMIT_BAL": (10_000, 2_000_000),
    "AGE": (18, 100),
    **{c: (-2, 8) for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]},
}

EXPECTED_COLUMNS = RAW_FEATURES + [TARGET]


def _is_numeric(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and pd.api.types.is_numeric_dtype(df[column])


def validate_schema(df: pd.DataFrame) -> List[str]:
    """Level 1 — are the expected columns present, with usable types?"""
    errors = []

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")

    non_numeric = [
        c for c in EXPECTED_COLUMNS
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        errors.append(f"non-numeric columns: {non_numeric}")

    return errors


def validate_statistics(df: pd.DataFrame) -> List[str]:
    """Level 2 — is the shape of the data what training assumes?"""
    errors = []

    if len(df) < MIN_ROWS:
        errors.append(f"too few rows: {len(df)} < {MIN_ROWS}")

    if len(df) > 0:
        missing_fraction = df.isna().mean()
        for column, fraction in missing_fraction[missing_fraction > MAX_MISSING_FRACTION].items():
            errors.append(
                f"column {column} is {fraction:.1%} missing (limit {MAX_MISSING_FRACTION:.1%})"
            )

    if _is_numeric(df, TARGET) and df[TARGET].notna().any():
        rate = float(df[TARGET].mean())
        if not MIN_POSITIVE_RATE <= rate <= MAX_POSITIVE_RATE:
            errors.append(
                f"target positive rate {rate:.3f} outside "
                f"[{MIN_POSITIVE_RATE}, {MAX_POSITIVE_RATE}]"
            )

    return errors


def validate_semantics(df: pd.DataFrame) -> List[str]:
    """Level 3 — do the values mean what the business says they mean?"""
    errors = []

    for column, allowed in DOMAINS.items():
        if column not in df.columns:
            continue
        values = df[column].dropna()
        invalid = values[~values.isin(allowed)]
        if not invalid.empty:
            errors.append(
                f"{column} has {len(invalid)} values outside {sorted(allowed)}: "
                f"{sorted(invalid.unique().tolist())[:5]}"
            )

    for column, (low, high) in RANGES.items():
        if not _is_numeric(df, column):
            continue
        values = df[column].dropna()
        n_outside = int((~values.between(low, high)).sum())
        if n_outside:
            errors.append(f"{column} has {n_outside} values outside [{low}, {high}]")

    for column in (c for c in df.columns if c.startswith("PAY_AMT")):
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        n_negative = int((df[column] < 0).sum())
        if n_negative:
            errors.append(f"{column} has {n_negative} negative values")

    return errors


def validate_dataset(df: pd.DataFrame, raise_on_error: bool = True) -> Dict[str, Any]:
    """Run all three levels and return a report."""
    schema_errors = validate_schema(df)
    statistical_errors = validate_statistics(df)
    semantic_errors = validate_semantics(df)
    errors = schema_errors + statistical_errors + semantic_errors

    report = {
        "passed": not errors,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "schema_errors": schema_errors,
        "statistical_errors": statistical_errors,
        "semantic_errors": semantic_errors,
        "n_errors": len(errors),
    }

    for error in errors:
        logger.error("Validation: %s", error)

    if errors and raise_on_error:
        raise DataValidationError(f"{len(errors)} validation error(s): " + "; ".join(errors))

    if not errors:
        logger.info("Validation passed: %s rows x %s columns", report["n_rows"], report["n_columns"])
    return report
