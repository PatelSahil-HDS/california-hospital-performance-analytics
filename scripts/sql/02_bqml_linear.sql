-- Linear regression in BigQuery ML predicting Performance_Rating.

CREATE OR REPLACE MODEL `your_project.your_dataset.hospital_linear`
OPTIONS(
  model_type = 'LINEAR_REG',
  input_label_cols = ['Performance_Rating'],
  data_split_method = 'RANDOM',
  data_split_eval_fraction = 0.2
) AS
SELECT
  Performance_Rating,
  Risk_Adjusted_Rate,
  Adverse_Rate,
  Hospital_Type,
  Region
FROM `your_project.your_dataset.hospital_features`;

-- Evaluation
SELECT *
FROM ML.EVALUATE(MODEL `your_project.your_dataset.hospital_linear`);

-- Feature importance via standardized weights
SELECT *
FROM ML.WEIGHTS(MODEL `your_project.your_dataset.hospital_linear`,
                STRUCT(TRUE AS standardize))
ORDER BY ABS(weight) DESC;
