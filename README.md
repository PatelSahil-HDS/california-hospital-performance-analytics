# Predicting Hospital Performance in California

This is the code, data, and write-up for a research project I did on hospital performance ratings in California. I used the 2016 Heart Failure inpatient cohort from the California Department of Health Care Access and Information (HCAI) — 137 hospitals, real public data, official OSHPD facility IDs — and fit both linear and logistic regression models to figure out what drives a hospital's quality rating.

The paper (PDF + DOCX) is in this repo if you want the full write-up. Everything below is the short version.

**Author:** Sahil Patel, Dept. of Health Data Science, Saint Louis University
**Contact:** patelsahils9099@gmail.com

---

## What I was trying to answer

1. Which hospital characteristics (type, region, risk-adjusted mortality, adverse event rate) actually predict the overall HCAI performance rating?
2. Can a simple logistic regression usefully classify hospitals as high vs. low performers (cutoff = 70) using only publicly available HCAI data?

## TL;DR results

Linear regression got an R² of 0.41 on the held-out test set, but cross-validated R² was unstable (mean 0.02, SD 0.20) — which is what you'd expect with n=137 and only seven features. Logistic regression hit 64.3% test accuracy and AUC of 0.77, with cross-validated accuracy at 70.8%.

Two findings held up across both models:

- **Risk-adjusted mortality rate** is by far the strongest negative predictor (linear β = −8.75; logit OR = 0.30).
- **Teaching hospitals** outperform General and Community facilities (mean rating 85.6 vs. 71.5 vs. 69.3). In the logistic model they're about twice as likely to be classified as high performers (OR = 2.01).

There's also a counterintuitive negative coefficient on Bay Area hospitals vs. the Central Valley reference. My read is that big academic centers in the Bay Area treat more complex Heart Failure cases, which depresses their risk-adjusted numbers even when their actual care is excellent. Werner & Bradlow (2006) make a similar argument about Hospital Compare.

## Data

Source: [California HCAI Inpatient Mortality Indicators](https://data.chhs.ca.gov/dataset/california-hospital-inpatient-mortality-rates-and-quality-ratings), Heart Failure cohort, FY 2016. Public data — no IRB or DUA required.

After dropping statewide aggregate rows and hospitals with missing Heart Failure entries, I ended up with 137 hospitals across 30 counties and 4 regions. The CSV (`California_Hospital_Real_Data.csv`) has these columns:

- `HOSPITAL`, `OSHPDID` — facility name and official OSHPD ID
- `Hospital_Type` — General / Community / Teaching
- `Region`, `County` — geographic
- `Condition`, `Year` — fixed to Heart Failure, 2016
- `Risk_Adjusted_Rate` — AHRQ PSI risk-adjusted mortality rate
- `Cases`, `Adverse_Events`, `Adverse_Rate`
- `Performance_Rating` — 0–100 composite
- `Hospital_Rating_Category` — Better / As Expected / Worse

I picked Heart Failure as the index condition because it has the broadest hospital coverage in the HCAI registry — most facilities report it.

## What I used

- **Python 3.10** — pandas, scikit-learn, matplotlib, seaborn
- **R 4.3** — tidyverse, ggplot2, caret, pROC
- **SQL** — Google BigQuery + BigQuery ML (parallel `LINEAR_REG` and `LOGISTIC_REG` models)
- **Dashboards** — Tableau Desktop, Power BI Desktop, Looker Studio (live BigQuery view)

I trained the same two models three times — once in scikit-learn, once in R via `lm()` / `glm()`, once in BigQuery ML — to sanity-check that I'd get the same directional results across stacks. I did.

## Methodology notes

A couple of decisions worth flagging:

- **County was dropped from the feature matrix.** 30 counties × n=137 means roughly one binary parameter per 4.6 observations, which would just blow up variance. Region (4 levels) keeps the geographic signal without overparameterizing.
- **Cases and Adverse_Events were collapsed into a ratio.** Keeping all three blew up the VIF past 10. Adverse_Rate = Adverse_Events / Cases keeps the information without the multicollinearity.
- **One-hot encoding** with Community (Hospital_Type) and Central Valley (Region) as the implicit references.
- **Standardized features** (z-score) before fitting, so coefficients are comparable.
- **80/20 split** + 5-fold CV. Logistic regression is L2-regularized, C=1.0.

## Caveats

I'd rather be upfront about the limitations than oversell the results:

- Cross-sectional, single year (2016), single condition (Heart Failure). Don't extrapolate.
- The feature set is genuinely thin. Staffing ratios, payer mix, EHR adoption, accreditation, community SES — none of those are in the HCAI registry. They almost certainly matter.
- n=137 is small. The low cross-validated R² on the linear model is the honest signal here, not the 0.41 test R².
- Linear and logistic both assume additive effects. Random forests / gradient boosting would probably do better and I plan to add them.
- The 70 cutoff for "High Performer" is the sample median. Convenient, but somewhat arbitrary.

## Repo layout

```
.
├── data/
│   └── California_Hospital_Real_Data.csv     # raw HCAI export
├── scripts/
│   ├── python/
│   │   ├── 01_load_and_clean.py              # cleaning + feature engineering
│   │   ├── 02_eda.py                         # plots + group summaries
│   │   ├── 03_linear_regression.py           # OLS + 5-fold CV
│   │   └── 04_logistic_regression.py         # L2 logit + AUC
│   ├── r/
│   │   ├── eda.R                             # ggplot mirror of EDA
│   │   └── models.R                          # lm() + glm() mirror
│   └── sql/
│       ├── 01_create_feature_view.sql        # BigQuery feature view
│       ├── 02_bqml_linear.sql                # BigQuery ML linear
│       └── 03_bqml_logistic.sql              # BigQuery ML logistic
├── paper/
│   ├── Research_Paper_Hospital_Analytics.pdf
│   └── Research_Paper_Hospital_Analytics.docx
├── requirements.txt
└── README.md
```

## Running it locally

```bash
git clone https://github.com/PatelSahil-HDS/california-hospital-performance-analytics.git
cd california-hospital-performance-analytics

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/python/01_load_and_clean.py
python scripts/python/02_eda.py
python scripts/python/03_linear_regression.py
python scripts/python/04_logistic_regression.py
```

R side:

```r
install.packages(c("tidyverse", "caret", "pROC", "corrplot"))
Rscript scripts/r/eda.R
Rscript scripts/r/models.R
```

For the BigQuery side, edit `your_project.your_dataset` in the SQL files and run them in the BigQuery console.

## What's next

I want to extend this in a few directions:

- Pull the 2016–2023 HCAI panel and treat it longitudinally
- Add AMI, Stroke, and Pneumonia cohorts so it isn't Heart Failure–only
- Swap in tree-based models (XGBoost, LightGBM) and compare against the linear baselines
- Look at whether CMS Hospital Compare and Leapfrog features add signal beyond HCAI

## Citation

```
Patel, S. (2025). Predicting Hospital Performance in California:
A Multi-Method Analytics Approach Using Linear and Logistic Regression.
Saint Louis University, Department of Health Data Science.
https://github.com/PatelSahil-HDS/california-hospital-performance-analytics
```

## License

Code is MIT (see `LICENSE`). The underlying HCAI dataset is governed by the California Health and Human Services Open Data Portal terms.

