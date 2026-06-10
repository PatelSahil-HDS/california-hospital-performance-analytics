"""
Load the HCAI 2016 Heart Failure dataset and do basic cleaning.

Reads:  data/California_Hospital_Real_Data.csv
Writes: data/hospital_clean.csv
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "California_Hospital_Real_Data.csv"
CLEAN = DATA_DIR / "hospital_clean.csv"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    print(f"Loaded {len(df)} rows from {RAW.name}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize categoricals
    for col in ["Hospital_Type", "Region", "County", "Hospital_Rating_Category"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    # Engineered features
    df["Adverse_Rate"] = df["Adverse_Events"] / df["Cases"]
    df["High_Performer"] = (df["Performance_Rating"] >= 70).astype(int)

    # Sanity checks
    assert df["Performance_Rating"].between(0, 100).all()
    assert df["Cases"].gt(0).all()
    assert df.isna().sum().sum() == 0, "Unexpected NA values"

    return df


def main() -> None:
    df = clean(load_raw())
    df.to_csv(CLEAN, index=False)
    print(f"Wrote {len(df)} rows -> {CLEAN.name}")
    print(df.describe(include="all").T.head(15))


if __name__ == "__main__":
    main()
