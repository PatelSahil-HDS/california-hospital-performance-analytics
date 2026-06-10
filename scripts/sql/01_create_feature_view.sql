-- BigQuery feature view for the HCAI Heart Failure cohort.
-- Replace `your_project.your_dataset` with your project/dataset.

CREATE OR REPLACE VIEW `your_project.your_dataset.hospital_features` AS
SELECT
  HOSPITAL,
  OSHPDID,
  Hospital_Type,
  Region,
  County,
  Risk_Adjusted_Rate,
  Cases,
  Adverse_Events,
  SAFE_DIVIDE(Adverse_Events, Cases)              AS Adverse_Rate,
  Performance_Rating,
  IF(Performance_Rating >= 70, 1, 0)              AS High_Performer
FROM `your_project.your_dataset.hospital_raw`
WHERE Condition = 'Heart Failure'
  AND Year = 2016
  AND HOSPITAL IS NOT NULL;
