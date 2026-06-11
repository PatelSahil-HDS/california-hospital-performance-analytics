# What I Learned Trying to Predict Hospital Quality with 137 Data Points

*A plain-English writeup of my California hospital analytics project. The full paper, code, and data are on [GitHub](https://github.com/PatelSahil-HDS/california-hospital-performance-analytics).*

---

I spent the last few months trying to answer what sounds like a simple question: can I predict how a California hospital is doing from publicly available data?

Spoiler: kind of. The interesting part is *why* the answer is "kind of."

---

## The setup

California's Department of Health Care Access and Information (HCAI) publishes hospital quality data every year. For Heart Failure, which is the condition with the broadest reporting coverage, I pulled the 2016 inpatient cohort — 137 hospitals across 30 counties.

Each hospital has:

- A **risk-adjusted mortality rate** (how often Heart Failure patients die, adjusted for how sick they were on admission)
- A **performance rating** from 0 to 100 (HCAI's composite quality score)
- Categorical info like hospital type (Community, General, Teaching) and region

The question I wanted to answer: **What predicts a hospital's performance rating?**

---

## What I built

Two models:

1. **Linear regression** to predict the actual rating (a number between 0 and 100)
2. **Logistic regression** to classify hospitals as "High Performer" (rating ≥ 70) vs. not

I built both in three different stacks — Python with scikit-learn, R with `glm()`, and Google BigQuery ML — partly to learn the tools, partly to verify the results held up across them. They did.

---

## What I found

Two things that should not surprise anyone in healthcare:

**Teaching hospitals do better.** Mean performance rating of 85.6 vs. 71.5 for General hospitals and 69.3 for Community. In the logistic model, Teaching hospitals are about twice as likely to be classified as High Performers (odds ratio 2.0). The literature has said this for years; the data agrees.

**Risk-adjusted mortality is the strongest single predictor.** A one-standard-deviation increase in risk-adjusted mortality drops the predicted performance rating by about 9 points, and reduces the odds of being a High Performer by ~70%. This is mostly tautological — HCAI partly *defines* the rating using mortality — but it confirms the model is at least internally consistent.

One thing that did surprise me:

**Bay Area hospitals score lower than the Central Valley reference**, even after controlling for hospital type. My best guess is that big academic centers in San Francisco and Oakland treat more complex Heart Failure cases — the kind that aren't fully captured by AHRQ's risk adjustment. So their numerator (mortality) is high relative to what the model expects, and they get penalized. Werner and Bradlow made a similar argument about Hospital Compare back in 2006. Worth digging into in future work.

---

## The honest part — what didn't work

This is the part I wish more analytics writeups covered.

The linear model got an R² of 0.41 on the held-out test set. Sounds decent. But the 5-fold cross-validated R² was 0.02, with high variance across folds (one fold even went negative).

**That gap is the whole story.**

What it means in plain English: the model can fit a specific train/test split moderately well, but if you change which hospitals fall into the training vs. test set, performance is all over the place. That's a sign the model is fragile — there isn't a stable underlying signal it's locking onto.

The logistic model did better. Test accuracy of 64%, AUC of 0.77, cross-validated accuracy of 70.8%. It's actually useful for ranking hospitals by their probability of being High Performers, even if individual classifications aren't reliable.

---

## Why the models were fragile

Three reasons, in roughly increasing importance:

**1. The sample is small.** 137 hospitals isn't many. With seven features that's roughly one parameter per 20 observations — workable but not generous.

**2. The features are thin.** HCAI's registry has hospital type, region, mortality rate, and case counts. That's it. The things that *actually* predict hospital quality in the literature — nursing-to-patient ratios, payer mix, EHR adoption, community socioeconomic status, accreditation — aren't in this dataset.

**3. Hospital performance is genuinely multifactorial.** Even with richer data, you'd expect the variance to be split across many small effects rather than dominated by a few big ones. The correlation matrix in my notebook shows this: no single feature has |r| above 0.42 with the outcome. That's the signal that no simple model is going to crush this problem.

---

## What I'd do differently next time

A few things on my list:

- **Pull the multi-year HCAI panel (2016–2023)** and model trends within hospitals over time, not just cross-sectional snapshots
- **Add other clinical conditions** — AMI, stroke, pneumonia — so I'm not over-fitting to Heart Failure's specific dynamics
- **Bring in CMS Hospital Compare or Leapfrog data** to enrich the feature set with staffing, safety culture, and patient experience metrics
- **Try non-linear models** — random forest and gradient boosting are good fits when you suspect interaction effects and small effect sizes
- **Look at outliers as case studies**, not noise. The two Bay Area hospitals that scored worst probably have stories worth understanding

---

## The bigger lesson

The most useful thing I learned isn't about hospitals — it's about evaluation.

A test R² of 0.41 with cross-validation of 0.02 is a model that *looks* okay in a presentation slide and falls apart in production. If I'd reported only the test R², the headline would have been "model explains 41% of the variation in hospital quality." That sentence is technically true and substantively misleading.

Cross-validation isn't a formality. It's the thing that tells you whether your model would survive contact with a different sample.

---

## Code, data, paper

Everything's open source:

- **Repo:** [github.com/PatelSahil-HDS/california-hospital-performance-analytics](https://github.com/PatelSahil-HDS/california-hospital-performance-analytics)
- **Paper (PDF):** in the `paper/` folder of the repo
- **Interactive demo:** Streamlit app in the repo (`streamlit run app.py`)

If you work in healthcare analytics or are using HCAI data for something, I'd love to compare notes. You can reach me at patelsahils9099@gmail.com or on [LinkedIn](https://www.linkedin.com/in/sahil-patel-hds).

---

*Sahil Patel is an MS Health Data Science candidate at Saint Louis University.*
