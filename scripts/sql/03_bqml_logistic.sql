-- Logistic regression in BigQuery ML classifying High Performers (Rating >= 70).

CREATE OR REPLACE MODEL `your_project.your_dataset.hospital_logit`
OPTIONS(
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['High_Performer'],
  data_split_method = 'RANDOM',
  data_split_eval_fraction = 0.2,
  auto_class_weights = TRUE
) AS
SELECT
  High_Performer,
  Risk_Adjusted_Rate,
  Adverse_Rate,
  Hospital_Type,
  Region
FROM `your_project.your_dataset.hospital_features`;

-- Confusion matrix + ROC metrics
SELECT *
FROM ML.EVALUATE(MODEL `your_project.your_dataset.hospital_logit`);

SELECT *
FROM ML.CONFUSION_MATRIX(MODEL `your_project.your_dataset.hospital_logit`,
  (SELECT * FROM `your_project.your_dataset.hospital_features`));

-- Score new rows
SELECT
  HOSPITAL,
  predicted_High_Performer,
  predicted_High_Performer_probs
FROM ML.PREDICT(MODEL `your_project.your_dataset.hospital_logit`,
  (SELECT * FROM `your_project.your_dataset.hospital_features`));
