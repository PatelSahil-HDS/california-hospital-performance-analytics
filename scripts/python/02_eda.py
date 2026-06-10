"""
Exploratory analysis: summary stats, group comparisons, correlation matrix.

Reads:  data/hospital_clean.csv
Writes: figures/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "hospital_clean.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def main() -> None:
    df = pd.read_csv(DATA)

    print("\n--- By Hospital Type ---")
    print(df.groupby("Hospital_Type")["Performance_Rating"].agg(["count", "mean", "std"]))

    print("\n--- By Region ---")
    print(df.groupby("Region")["Performance_Rating"].agg(["count", "mean", "std"]))

    # Distribution of the outcome
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Performance_Rating"], bins=20, ax=ax, kde=True)
    ax.set_title("Performance Rating distribution (n=%d)" % len(df))
    ax.set_xlabel("Performance Rating")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rating_distribution.png", dpi=150)

    # Boxplot by hospital type
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Hospital_Type", y="Performance_Rating",
                order=["Community", "General", "Teaching"], ax=ax)
    ax.set_title("Performance Rating by Hospital Type")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rating_by_type.png", dpi=150)

    # Scatter: risk-adjusted mortality vs rating
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=df, x="Risk_Adjusted_Rate", y="Performance_Rating",
                scatter_kws={"alpha": 0.6}, ax=ax)
    ax.set_title("Risk-Adjusted Mortality vs Performance Rating")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rate_vs_rating.png", dpi=150)

    # Correlation matrix on numeric features
    num_cols = ["Performance_Rating", "Risk_Adjusted_Rate", "Cases",
                "Adverse_Events", "Adverse_Rate"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Pearson correlations")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_matrix.png", dpi=150)

    print(f"\nWrote 4 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
