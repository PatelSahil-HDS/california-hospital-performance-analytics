"""
Multiple linear regression on Performance_Rating.

Reads:  data/hospital_clean.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[2] / "data" / "hospital_clean.csv"
SEED = 42


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    # Community and Central Valley are the implicit references
    type_dummies = pd.get_dummies(df["Hospital_Type"], prefix="Type", drop_first=False)
    region_dummies = pd.get_dummies(df["Region"], prefix="Region", drop_first=False)
    type_dummies = type_dummies.drop(columns=["Type_Community"], errors="ignore")
    region_dummies = region_dummies.drop(columns=["Region_Central Valley"], errors="ignore")

    feat = pd.concat(
        [df[["Risk_Adjusted_Rate", "Adverse_Rate"]], type_dummies, region_dummies],
        axis=1,
    ).astype(float)
    return feat, df["Performance_Rating"].astype(float), list(feat.columns)


def main() -> None:
    df = pd.read_csv(DATA)
    X, y, cols = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    model = LinearRegression().fit(Xtr, y_train)
    pred = model.predict(Xte)

    print(f"Test R²:   {r2_score(y_test, pred):.3f}")
    print(f"Test MAE:  {mean_absolute_error(y_test, pred):.2f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    for tr_idx, va_idx in cv.split(X):
        sc = StandardScaler().fit(X.iloc[tr_idx])
        m = LinearRegression().fit(sc.transform(X.iloc[tr_idx]), y.iloc[tr_idx])
        cv_scores.append(r2_score(y.iloc[va_idx], m.predict(sc.transform(X.iloc[va_idx]))))
    print(f"CV R² (5-fold): mean={np.mean(cv_scores):.3f}  sd={np.std(cv_scores):.3f}")
    print(f"  fold values: {[round(s, 3) for s in cv_scores]}")

    print("\nStandardized coefficients (sorted by |β|):")
    coef = pd.Series(model.coef_, index=cols).sort_values(key=abs, ascending=False)
    for name, val in coef.items():
        print(f"  {name:25s}  {val:+8.3f}")


if __name__ == "__main__":
    main()
