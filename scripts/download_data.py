"""
Download the real UCI "Default of Credit Card Clients" dataset (optional).

The lab ships with a generated dataset in data/credit_default.csv so it runs
with no network access at all. This script replaces that file with the real
one when you do have network access — the schema is identical, so nothing
else in the lab changes.

    Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques
    for the predictive accuracy of probability of default of credit card clients.
    Expert Systems with Applications, 36(2), 2473-2480.
    UCI Machine Learning Repository, dataset 350. Licence: CC BY 4.0.

Usage:
    pip install ucimlrepo openpyxl
    python scripts/download_data.py
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = BASE_DIR / "data" / "credit_default.csv"

UCI_XLS = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

RENAME = {"PAY_1": "PAY_0", "default payment next month": "default_payment_next_month",
          "Y": "default_payment_next_month"}


def main() -> None:
    print("Fetching UCI dataset 350 ...")
    try:
        from ucimlrepo import fetch_ucirepo

        repo = fetch_ucirepo(id=350)
        df = repo.data.features.join(repo.data.targets)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  Could not reach the UCI repository: {exc}")
        print("\n  Your network probably blocks archive.ics.uci.edu.")
        print("  Download the file manually instead:")
        print(f"    {UCI_XLS}")
        print("  Unzip it, then convert the .xls to CSV with:")
        print("    python -c \"import pandas as pd; "
              "pd.read_excel('default of credit card clients.xls', header=1)"
              f".to_csv('{OUT_PATH.name}', index=False)\"")
        print(f"\n  Place the result at: {OUT_PATH}")
        print("  The lab works fine on the generated dataset until then.")
        raise SystemExit(1)

    df = df.rename(columns=RENAME)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    expected = 24
    if df.shape[1] != expected:
        raise SystemExit(f"Unexpected shape {df.shape}; expected {expected} columns")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} columns -> {OUT_PATH}")
    print(f"Default rate: {df['default_payment_next_month'].mean():.1%}")
    print("\nRetrain to pick up the new data:  python scripts/train_model.py")


if __name__ == "__main__":
    main()
