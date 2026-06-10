"""
Binary logistic regression classifying High Performers (Rating >= 70).

Reads:  data/hospital_clean.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[2] / "data" / "hospital_clean.csv"
SEED = 42


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    type_dummies = pd.get_dummies(df["Hospital_Type"], prefix="Type", drop_first=False)
    region_dummies = pd.get_dummies(df["Region"], prefix="Region", drop_first=False)
    type_dummies = type_dummies.drop(columns=["Type_Community"], errors="ignore")
    region_dummies = region_dummies.drop(columns=["Region_Central Valley"], errors="ignore")

    feat = pd.concat(
        [df[["Risk_Adjusted_Rate", "Adverse_Rate"]], type_dummies, region_dummies],
        axis=1,
    ).astype(float)
    return feat, df["High_Performer"].astype(int), list(feat.columns)


def main() -> None:
    df = pd.read_csv(DATA)
    X, y, cols = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(Xtr, y_train)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]

    print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
    print(f"Test AUC:      {roc_auc_score(y_test, proba):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["Low", "High"]))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, pred))

    cv_acc = cross_val_score(
        LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        StandardScaler().fit_transform(X), y, cv=5, scoring="accuracy",
    )
    print(f"\nCV accuracy (5-fold): mean={cv_acc.mean():.3f}  sd={cv_acc.std():.3f}")

    print("\nStandardized coefficients and odds ratios:")
    print(f"{'Feature':25s} {'β':>10s} {'OR':>10s}")
    coefs = pd.Series(model.coef_[0], index=cols).sort_values(key=abs, ascending=False)
    for name, beta in coefs.items():
        print(f"{name:25s} {beta:+10.3f} {np.exp(beta):>10.3f}")


if __name__ == "__main__":
    main()
