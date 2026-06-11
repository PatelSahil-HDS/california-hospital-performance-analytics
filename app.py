"""
Interactive Streamlit demo of the California Hospital Performance model.

Lets a user enter hospital characteristics and see both the predicted
performance rating (linear regression) and the probability of being
classified as a High Performer (logistic regression).

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud:
    https://streamlit.io/cloud
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "California_Hospital_Real_Data.csv"

st.set_page_config(
    page_title="California Hospital Performance Predictor",
    page_icon="🏥",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Data + model (cached so we don't refit on every interaction)
# ----------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)
    for col in ["Hospital_Type", "Region", "County", "Hospital_Rating_Category"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    df["Adverse_Rate"] = df["Adverse_Events"] / df["Cases"]
    df["High_Performer"] = (df["Performance_Rating"] >= 70).astype(int)
    return df


@st.cache_resource
def fit_models(df: pd.DataFrame):
    """Fit the same models described in the paper, return a bundle."""
    type_dummies = pd.get_dummies(df["Hospital_Type"], prefix="Type")
    region_dummies = pd.get_dummies(df["Region"], prefix="Region")
    type_dummies = type_dummies.drop(columns=["Type_Community"], errors="ignore")
    region_dummies = region_dummies.drop(columns=["Region_Central Valley"], errors="ignore")

    feature_cols = ["Risk_Adjusted_Rate", "Adverse_Rate"] + \
                   list(type_dummies.columns) + list(region_dummies.columns)
    X = pd.concat([df[["Risk_Adjusted_Rate", "Adverse_Rate"]], type_dummies, region_dummies],
                  axis=1).astype(float)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    lin = LinearRegression().fit(Xs, df["Performance_Rating"])
    log = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(Xs, df["High_Performer"])

    return {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "linear": lin,
        "logistic": log,
    }


df = load_data()
bundle = fit_models(df)


# ----------------------------------------------------------------------------
# Page header
# ----------------------------------------------------------------------------

st.title("🏥 California Hospital Performance Predictor")
st.markdown(
    "Predicts a hospital's HCAI performance rating and whether it'll be classified "
    "as a High Performer (rating ≥ 70), using the same linear and logistic regression "
    "models from [the research paper](https://github.com/PatelSahil-HDS/california-hospital-performance-analytics)."
)

st.caption(
    "Data: California HCAI Inpatient Mortality Indicators, Heart Failure cohort, FY 2016 (n = 137). "
    "Models trained on the full sample for this demo."
)


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------

left, right = st.columns([1, 1])

with left:
    st.subheader("Hospital characteristics")

    hospital_type = st.selectbox(
        "Hospital type",
        ["Community", "General", "Teaching"],
        index=1,
        help="Teaching hospitals tend to score higher in the HCAI dataset.",
    )

    region = st.selectbox(
        "Region",
        ["Central Valley", "Bay Area", "Northern California", "Southern California"],
        index=1,
    )

    risk_rate = st.slider(
        "Risk-adjusted Heart Failure mortality rate (%)",
        min_value=0.0, max_value=20.0, value=3.0, step=0.1,
        help="AHRQ PSI risk-adjusted inpatient mortality rate. Statewide mean ≈ 3.1%.",
    )

    cases = st.number_input(
        "Heart Failure cases (annual)",
        min_value=10, max_value=2500, value=400, step=10,
    )

    adverse_events = st.number_input(
        "Adverse events (annual)",
        min_value=0, max_value=100, value=12, step=1,
    )

    adverse_rate = adverse_events / max(cases, 1)
    st.caption(f"Adverse rate = {adverse_rate:.3%}")


# ----------------------------------------------------------------------------
# Feature vector → predictions
# ----------------------------------------------------------------------------

row = {col: 0.0 for col in bundle["feature_cols"]}
row["Risk_Adjusted_Rate"] = risk_rate
row["Adverse_Rate"] = adverse_rate
if hospital_type != "Community":
    row[f"Type_{hospital_type}"] = 1.0
if region != "Central Valley":
    row[f"Region_{region}"] = 1.0

X_user = pd.DataFrame([row], columns=bundle["feature_cols"])
Xs_user = bundle["scaler"].transform(X_user)

predicted_rating = float(bundle["linear"].predict(Xs_user)[0])
predicted_rating = max(0.0, min(100.0, predicted_rating))
prob_high = float(bundle["logistic"].predict_proba(Xs_user)[0, 1])


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

with right:
    st.subheader("Model predictions")

    st.metric(
        "Predicted Performance Rating",
        f"{predicted_rating:.1f} / 100",
        delta=f"{predicted_rating - 71.75:+.1f} vs. CA average (71.75)",
        help="Linear regression. The 0–100 scale is the HCAI composite quality score.",
    )

    st.metric(
        "Probability of being a High Performer",
        f"{prob_high:.1%}",
        delta="threshold = Rating ≥ 70",
        help="Logistic regression. Sample baseline: 56.2% of CA hospitals are High Performers.",
    )

    if prob_high >= 0.75:
        st.success("Strong High Performer signal.")
    elif prob_high >= 0.5:
        st.info("Likely High Performer, but uncertain.")
    elif prob_high >= 0.25:
        st.warning("Borderline. Closer to Low Performer.")
    else:
        st.error("Strong Low Performer signal.")

st.divider()


# ----------------------------------------------------------------------------
# Context — where this hospital would land vs. real data
# ----------------------------------------------------------------------------

st.subheader("Where this profile lands vs. real California hospitals")

import altair as alt

scatter = alt.Chart(df).mark_circle(opacity=0.5, color="#4C78A8").encode(
    x=alt.X("Risk_Adjusted_Rate", title="Risk-Adjusted Mortality Rate (%)"),
    y=alt.Y("Performance_Rating", title="Performance Rating"),
    tooltip=["HOSPITAL", "Hospital_Type", "Region",
             "Performance_Rating", "Risk_Adjusted_Rate"],
).properties(height=400)

user_point = alt.Chart(pd.DataFrame([{
    "Risk_Adjusted_Rate": risk_rate,
    "Performance_Rating": predicted_rating,
}])).mark_circle(size=250, color="#E45756").encode(
    x="Risk_Adjusted_Rate",
    y="Performance_Rating",
)

st.altair_chart(scatter + user_point, use_container_width=True)
st.caption("Blue points: 137 California hospitals. Red point: the profile you entered.")


# ----------------------------------------------------------------------------
# Caveats — keep the user honest
# ----------------------------------------------------------------------------

with st.expander("Caveats — please read before drawing conclusions"):
    st.markdown(
        """
- This is a demonstration of the paper's regression models, not a clinical
  decision tool. **Do not use it to evaluate real hospitals.**
- The linear model's cross-validated R² is only 0.02 (with high variance).
  Test-set R² of 0.41 looks better but reflects a small held-out partition.
- Predictions extrapolate badly outside the training range — especially for
  Risk-Adjusted Rates above ~10% or for hospital types not represented in
  the 137-hospital sample.
- HCAI 2016 data is cross-sectional and limited to the Heart Failure cohort.
  Findings won't generalize to other conditions or years.
        """
    )

st.divider()
st.markdown(
    "Source code & paper: "
    "[github.com/PatelSahil-HDS/california-hospital-performance-analytics]"
    "(https://github.com/PatelSahil-HDS/california-hospital-performance-analytics)"
)
