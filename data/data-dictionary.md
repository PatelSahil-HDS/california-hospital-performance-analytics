# Data Dictionary — California HCAI Heart Failure Cohort

Source: California Department of Health Care Access and Information (HCAI), Inpatient Mortality Indicators registry, Heart Failure cohort, fiscal year 2016. Public dataset, retrieved from the [California Health and Human Services Open Data Portal](https://data.chhs.ca.gov/dataset/california-hospital-inpatient-mortality-rates-and-quality-ratings).

**Rows:** 137 hospitals, one row per hospital
**Columns:** 13 raw + 2 engineered

| Column | Type | Description | Range / Values | Source |
|---|---|---|---|---|
| `HOSPITAL` | string | Hospital name as registered with HCAI. Title case. | 137 unique values | HCAI |
| `Hospital_Type` | categorical | Facility classification. Used as a predictor. Community is the implicit reference category in regression models. | `General` (62.0%), `Community` (31.4%), `Teaching` (6.6%) | HCAI |
| `Region` | categorical | Geographic region in California. Central Valley is the implicit reference category. | `Southern California` (36.5%), `Bay Area` (28.5%), `Central Valley` (18.2%), `Northern California` (16.8%) | HCAI |
| `County` | categorical | California county. **Dropped from the feature matrix** (30 unique values with n=137 would severely overparameterize). Retained for reference. | 30 unique California counties | HCAI |
| `Condition` | categorical | Clinical condition. Fixed to "Heart Failure" in this dataset. | `Heart Failure` only | HCAI |
| `Year` | integer | Fiscal year of the cohort. | `2016` only | HCAI |
| `Performance_Rating` | continuous | **Primary outcome for the linear regression model.** Composite 0–100 quality score derived from HCAI's rating category and risk-adjusted mortality rate. Higher is better. | 27.6 – 99.6 (mean 71.75, SD 13.61, median 72.9) | Derived from HCAI rating category |
| `Risk_Adjusted_Rate` | continuous | AHRQ Patient Safety Indicator risk-adjusted inpatient mortality rate for Heart Failure cases. Standardized against statewide severity benchmarks. Lower is better. Strongest single predictor in both models. | 0.0 – 18.9 percent (mean 3.12, SD 1.87) | HCAI / AHRQ PSI |
| `Cases` | integer | Number of Heart Failure inpatient cases reported by the hospital in FY 2016. | 11 – 2,103 (mean 459.2, SD 294.5) | HCAI |
| `Adverse_Events` | integer | Count of adverse inpatient mortality events for Heart Failure. **Not used directly** in models due to multicollinearity with Cases (VIF > 10); used to derive `Adverse_Rate`. | 0 – 42 (mean 14.24, SD 7.98) | HCAI |
| `Hospital_Rating_Category` | categorical | HCAI's categorical quality classification. | `As Expected` (69.3%), `Better` (24.1%), `Worse` (6.6%) | HCAI |
| `OSHPDID` | integer | Official Office of Statewide Health Planning and Development facility identifier. Persistent across years — useful for joining HCAI data across cohorts. | 9-digit numeric ID | HCAI / OSHPD |

## Engineered features

| Column | Type | Definition | Range | Why it exists |
|---|---|---|---|---|
| `Adverse_Rate` | continuous | `Adverse_Events / Cases` | 0.000 – 0.190 (mean 0.034, SD 0.019) | Normalized adverse burden. Replaces the raw counts in the regression feature matrix to avoid multicollinearity. |
| `High_Performer` | binary | `1` if `Performance_Rating >= 70` else `0` | `0` (43.8%) or `1` (56.2%) | **Primary outcome for the logistic regression classifier.** Threshold of 70 is the empirical sample median. |

## Notes on data quality

- **No missing values** across any of the 137 analytical records (verified in `scripts/python/01_load_and_clean.py`).
- **No duplicate hospital records.**
- HCAI excludes hospitals with insufficient Heart Failure case volume — the 137-hospital sample is already filtered to facilities with enough cases for risk adjustment to be meaningful.
- Statewide aggregate rows were dropped before analysis. Only hospital-level rows are included.
- All categorical variables were standardized to title case during cleaning.

## What's intentionally NOT in this dataset

These would be useful predictors but aren't in the HCAI Inpatient Mortality registry:

- Nursing-to-patient staffing ratios
- Physician board certification rates
- EHR adoption / Meaningful Use stage
- Payer mix (Medicare / Medicaid / commercial / uninsured percentages)
- Bed count or hospital size
- Teaching intensity (residents per bed)
- Community socioeconomic context (county-level income, education, insurance coverage)
- Accreditation status (Joint Commission, Magnet)

The low cross-validated R² of the linear model is most likely driven by the absence of these features, not by methodological issues. See section 5.4 of the paper for a fuller discussion.
