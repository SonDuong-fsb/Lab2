"""
Generate the DDM501 credit-default dataset.

Schema is identical to the UCI "Default of Credit Card Clients" dataset
(Yeh & Lien, 2009 - UCI id 350), so the real file is a drop-in replacement:
same column names, same value ranges, same dtypes.

Usage:
    python scripts/make_dataset.py            # writes data/credit_default.csv
    python scripts/make_dataset.py --rows 30000 --seed 501

Columns
-------
LIMIT_BAL                    credit limit (NT dollars)
SEX                          1 = male, 2 = female
EDUCATION                    1 = graduate school, 2 = university, 3 = high school, 4 = others
MARRIAGE                     1 = married, 2 = single, 3 = others
AGE                          age in years
PAY_0, PAY_2 .. PAY_6        repayment status, months t-1 .. t-6
                             -2 = no consumption, -1 = paid in full, 0 = revolving credit,
                              1..8 = months of payment delay
BILL_AMT1 .. BILL_AMT6       bill statement amount, months t-1 .. t-6
PAY_AMT1 .. PAY_AMT6         amount paid, months t-1 .. t-6
default_payment_next_month   1 = defaults next month, 0 = does not
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
TARGET = "default_payment_next_month"

COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_COLS
    + BILL_COLS
    + PAYAMT_COLS
    + [TARGET]
)


def generate(n_rows: int = 30_000, seed: int = 501) -> pd.DataFrame:
    """Build a credit-default dataset with realistic structure."""
    rng = np.random.default_rng(seed)

    # --- demographics -------------------------------------------------------
    sex = rng.choice([1, 2], size=n_rows, p=[0.396, 0.604])
    education = rng.choice([1, 2, 3, 4], size=n_rows, p=[0.353, 0.468, 0.164, 0.015])
    marriage = rng.choice([1, 2, 3], size=n_rows, p=[0.455, 0.532, 0.013])
    age = np.clip(rng.gamma(shape=2.8, scale=5.2, size=n_rows) + 21, 21, 79).astype(int)

    # --- latent creditworthiness -------------------------------------------
    # Higher score = better customer. Drives limit, repayment history and default.
    score = (
        rng.normal(0, 1, n_rows)
        + 0.30 * (education == 1)
        - 0.25 * (education == 3)
        + 0.012 * (age - 35)
        + 0.50 * (sex == 2)          # small demographic gap, on purpose:
    )                                # Lab 4 fairness analysis needs something to find

    # --- credit limit -------------------------------------------------------
    limit_bal = np.exp(rng.normal(11.4, 0.62, n_rows) + 0.30 * score)
    limit_bal = np.clip(np.round(limit_bal, -4), 10_000, 1_000_000)

    # --- repayment status: autoregressive across the six months -------------
    pay = np.zeros((n_rows, 6), dtype=int)
    latent = -0.9 * score + rng.normal(0, 0.8, n_rows)          # month t-6
    for m in range(5, -1, -1):                                   # t-6 .. t-1
        latent = 0.72 * latent + rng.normal(0, 0.62, n_rows)
        col = np.full(n_rows, -1)
        col = np.where(latent > -0.78, 0, col)                   # revolving credit
        col = np.where(latent > 0.70, 1, col)                    # 1 month late
        col = np.where(latent > 1.34, 2, col)
        col = np.where(latent > 2.05, 3, col)
        col = np.where(latent > 2.55, rng.integers(4, 9, n_rows), col)
        no_use = rng.random(n_rows) < 0.092
        col = np.where(no_use, -2, col)
        pay[:, m] = col

    # --- bills and payments -------------------------------------------------
    utilisation = np.clip(rng.beta(1.6, 3.1, n_rows) + 0.12 * (pay[:, 0] > 0), 0, 1.25)
    bills = np.zeros((n_rows, 6))
    base_bill = limit_bal * utilisation
    for m in range(6):
        drift = rng.normal(1.0, 0.14, n_rows) * (1 - 0.02 * m)
        bills[:, m] = np.maximum(base_bill * drift, 0)
    bills = np.where(pay == -2, 0, bills)
    bills = np.round(bills).astype(int)

    pay_amt = np.zeros((n_rows, 6))
    for m in range(6):
        # Customers in arrears pay a much smaller share of the statement.
        share = np.where(
            pay[:, m] <= 0,
            rng.beta(2.4, 1.5, n_rows),
            rng.beta(1.0, 9.0, n_rows),
        )
        pay_amt[:, m] = bills[:, m] * share
    pay_amt = np.round(np.maximum(pay_amt, 0)).astype(int)

    # --- target -------------------------------------------------------------
    pay_ratio = pay_amt[:, 0] / np.maximum(bills[:, 0], 1)
    logit = (
        -1.14
        + 0.78 * np.clip(pay[:, 0], -1, 8)
        + 0.24 * np.clip(pay[:, 1], -1, 8)
        + 0.11 * np.clip(pay[:, 2], -1, 8)
        + 0.85 * np.clip(utilisation, 0, 1.25)
        - 0.95 * np.clip(pay_ratio, 0, 1.5)
        - 0.32 * score
        - 0.010 * (age - 35)
    )
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.random(n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "LIMIT_BAL": limit_bal.astype(int),
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            **{c: pay[:, i] for i, c in enumerate(PAY_COLS)},
            **{c: bills[:, i] for i, c in enumerate(BILL_COLS)},
            **{c: pay_amt[:, i] for i, c in enumerate(PAYAMT_COLS)},
            TARGET: target,
        }
    )
    return df[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "data" / "credit_default.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate(args.rows, args.seed)
    df.to_csv(out, index=False)

    rate = df[TARGET].mean()
    print(f"Wrote {len(df):,} rows x {df.shape[1]} columns -> {out}")
    print(f"Default rate: {rate:.1%}")
    print(f"File size:    {out.stat().st_size / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
