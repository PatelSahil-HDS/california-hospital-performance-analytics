# Predicting Hospital Performance in California

**A Multi-Method Analytics Approach Using Linear and Logistic Regression**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.3.0-276DC3.svg)](https://www.r-project.org/)
[![BigQuery](https://img.shields.io/badge/Google-BigQuery%20ML-4285F4.svg)](https://cloud.google.com/bigquery)
[![Tableau](https://img.shields.io/badge/Tableau-Public-E97627.svg)](https://public.tableau.com/app/profile/sahil.patel)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the full analytical pipeline, dataset, and research paper for an end-to-end healthcare analytics study examining hospital performance across California. Using **137 California hospitals** from the **California Department of Health Care Access and Information (HCAI) 2016 Heart Failure inpatient cohort**, this project applies linear and logistic regression to identify the institutional drivers of hospital quality ratings.

The project demonstrates a **production-grade, multi-tool analytics workflow** integrating Python, R, SQL (BigQuery ML), and business intelligence platforms (Tableau, Power BI, Looker Studio) — all anchored in real, publicly available government health data.

> **Author:** Sahil Patel | Department of Health Data Science, Saint Louis University
> **Contact:** patelsahils9099@gmail.com

---

## Research Questions

- **RQ1:** Which hospital-level characteristics (type, region, risk-adjusted mortality rate, adverse event rate) are most strongly associated with overall performance rating?
- **RQ2:** To what extent can a logistic regression model accurately classify hospitals as high performers (Performance Rating ≥ 70) versus low performers using publicly available HCAI clinical data?

---

## Key Results

| Model | Metric | Value |
|-------|--------|-------|
| Linear Regression | Test R² | **0.41** |
| Linear Regression | MAE / RMSE | 9.16 / 10.94 |
| Logistic Regression | Test Accuracy | **64.3%** |
| Logistic Regression | AUC-ROC | **0.77** |
| Logistic Regression | CV Accuracy (5-fold) | **70.8%** (SD 6.9%) |

### Top Predictors
- **Risk-Adjusted Mortality Rate** — strongest *negative* predictor (β = −8.75; OR = 0.30)
- **Teaching Hospital Type** — strongest *positive* predictor (OR = 2.01)
- Teaching hospitals score **85.57** on average vs **71.51** (General) and **69.32** (Community)

---

## Dataset

**Source:** [California HCAI Inpatient Mortality Indicators](https://data.chhs.ca.gov/dataset/california-hospital-inpatient-mortality-rates-and-quality-ratings) — Heart Failure cohort, 2016

| Variable | Type | Description |
|----------|------|-------------|
| `HOSPITAL` | Text | Hospital name |
| `Hospital_Type` | Categorical | General / Community / Teaching |
| `Region` | Categorical | Bay Area / Southern CA / Central Valley / Northern CA |
| `County` | Categorical | 30 California counties |
| `Condition` | Categorical | Heart Failure |
| `Year` | Integer | 2016 |
| `Performance_Rating` | Continuous (0–100) | Composite quality score |
| `Risk_Adjusted_Rate` | Continuous | AHRQ PSI risk-adjusted mortality rate |
| `Cases` | Integer | Inpatient Heart Failure cases |
| `Adverse_Events` | Integer | Number of adverse events |
| `Adverse_Rate` | Engineered | Adverse_Events / Cases |
| `Hospital_Rating_Category` | Categorical | Better / As Expected / Worse |
| `OSHPDID` | Integer | Official OSHPD facility identifier |

**Sample size:** 137 hospitals across 30 counties and 4 regions.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Languages** | Python 3.10, R 4.3.0, SQL (Standard SQL) |
| **Python Libraries** | pandas, scikit-learn, matplotlib, seaborn |
| **R Libraries** | tidyverse, ggplot2, caret, pROC |
| **Cloud / Database** | Google BigQuery, BigQuery ML |
| **BI / Visualization** | Tableau Desktop, Power BI Desktop, Looker Studio |
| **Models** | Multiple Linear Regression, L2-Regularized Logistic Regression |

---

## Repository Structure

```
Research_Paper_Hospital_Analytics/
├── data/
│   └── California_Hospital_Real_Data.csv     # 137-hospital HCAI dataset
├── notebooks/
│   ├── 01_eda.ipynb                          # Exploratory data analysis
│   ├── 02_linear_regression.ipynb            # Linear model training & evaluation
│   └── 03_logistic_regression.ipynb          # Classifier training & evaluation
├── scripts/
│   ├── python/
│   │   ├── data_cleaning.py
│   │   ├── feature_engineering.py
│   │   ├── linear_model.py
│   │   └── logistic_model.py
│   ├── r/
│   │   ├── eda.R
│   │   ├── lm_model.R
│   │   └── glm_model.R
│   └── sql/
│       ├── bigquery_features.sql
│       ├── bqml_linear.sql
│       └── bqml_logistic.sql
├── dashboards/
│   ├── tableau/
│   ├── powerbi/
│   └── looker_studio/
├── paper/
│   ├── Research_Paper_Hospital_Analytics.pdf
│   └── Research_Paper_Hospital_Analytics.docx
└── README.md
```

---

## Methodology

1. **Data Acquisition** — HCAI Open Data Portal, 2016 Heart Failure cohort
2. **Cleaning & Feature Engineering**
   - One-hot encoding of `Hospital_Type` and `Region`
   - Engineered `Adverse_Rate` = `Adverse_Events` / `Cases`
   - Engineered `High_Performer` binary (Rating ≥ 70)
   - County excluded to avoid overparameterization (30 levels for n = 137)
3. **Exploratory Data Analysis** — Python (matplotlib, seaborn) + R (ggplot2)
4. **Modeling**
   - Linear regression with `Performance_Rating` outcome
   - Logistic regression with `High_Performer` outcome (L2 regularization, C = 1.0)
   - Standardized features (`StandardScaler`)
   - 80/20 train/test split + 5-fold cross-validation
5. **Cloud Replication** — Parallel models in BigQuery ML (`LINEAR_REG`, `LOGISTIC_REG`)
6. **Dashboards** — Tableau Public, Power BI, Looker Studio (live BigQuery view)

---

## Quick Start

### Clone the repository
```bash
git clone https://github.com/patelsahils9099-alt/Research_Paper_Hospital_Analytics.git
cd Research_Paper_Hospital_Analytics
```

### Python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook notebooks/01_eda.ipynb
```

### R environment
```r
install.packages(c("tidyverse", "ggplot2", "caret", "pROC", "corrplot"))
source("scripts/r/eda.R")
```

### BigQuery ML
```sql
-- See scripts/sql/bqml_logistic.sql
CREATE OR REPLACE MODEL `project.dataset.hospital_logit`
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['High_Performer']) AS
SELECT * FROM `project.dataset.hospital_features`;
```

---

## Key Findings

1. **Risk-Adjusted Mortality Rate** is the dominant negative predictor — a 1 SD increase corresponds to a ~70% reduction in the odds of being classified a High Performer.
2. **Teaching hospitals** are ~2× more likely to be High Performers than Community hospitals — consistent with the academic medical centre literature.
3. **Northern California** hospitals show systematically lower performance (mean rating 65.33) — driven in part by smaller rural facilities with higher mortality rates.
4. The model achieves **AUC = 0.77**, indicating strong rank-discrimination of hospital quality from a small set of administrative features.
5. Low cross-validated R² (0.02) confirms the **limits of administrative-only feature sets** — staffing ratios, payer mix, and EHR adoption are needed for robust generalization.

---

## Limitations

- Cross-sectional 2016 data; not yet longitudinal
- Limited to the Heart Failure inpatient cohort
- Narrow feature set (no staffing, payer mix, accreditation, or SES variables)
- n = 137 limits statistical power and CV stability
- Linear/logistic models assume additive relationships — ensemble methods may improve fit

---

## Roadmap

- [ ] Extend to 2016–2023 HCAI multi-year panel
- [ ] Add ensemble models (Random Forest, XGBoost, LightGBM)
- [ ] Incorporate additional clinical conditions (AMI, Stroke, Pneumonia)
- [ ] Integrate CMS Hospital Compare and Leapfrog Group features
- [ ] Deploy Streamlit / Flask scoring API backed by BigQuery ML

---

## Citation

If you use this work, please cite:

```bibtex
@misc{patel2025californiahospital,
  author       = {Patel, Sahil},
  title        = {Predicting Hospital Performance in California:
                  A Multi-Method Analytics Approach Using Linear and Logistic Regression},
  year         = {2025},
  institution  = {Saint Louis University, Department of Health Data Science},
  howpublished = {\url{https://github.com/patelsahils9099-alt}}
}
```

---

## License

This project is released under the [MIT License](LICENSE). The underlying HCAI dataset is published by the California Department of Health Care Access and Information under the California Health and Human Services Open Data terms.

---

## Author

**Sahil Patel**
Department of Health Data Science, Saint Louis University
- Email: patelsahils9099@gmail.com
- GitHub: [@patelsahils9099-alt](https://github.com/patelsahils9099-alt)
- Tableau Public: [public.tableau.com/app/profile/sahil.patel](https://public.tableau.com/app/profile/sahil.patel)

---

## Acknowledgments

- California Department of Health Care Access and Information (HCAI) for the open-data hospital quality registry
- AHRQ Patient Safety Indicator (PSI) software for risk-adjustment methodology
- Saint Louis University Department of Health Data Science
