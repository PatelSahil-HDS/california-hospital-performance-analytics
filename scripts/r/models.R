# Linear + logistic regression in R — mirrors the Python pipeline.
# Run from project root: Rscript scripts/r/models.R

library(tidyverse)
library(caret)
library(pROC)

df <- read_csv("data/hospital_clean.csv", show_col_types = FALSE)

df <- df |>
  mutate(Hospital_Type = factor(Hospital_Type,
                                levels = c("Community", "General", "Teaching")),
         Region = factor(Region,
                         levels = c("Central Valley", "Bay Area",
                                    "Northern California", "Southern California")))

set.seed(42)
idx <- createDataPartition(df$Performance_Rating, p = 0.8, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

# --- Linear regression -------------------------------------------------------
lm_fit <- lm(Performance_Rating ~ Risk_Adjusted_Rate + Adverse_Rate +
               Hospital_Type + Region,
             data = train)
cat("\n--- Linear regression summary ---\n")
print(summary(lm_fit))

lm_pred <- predict(lm_fit, newdata = test)
cat(sprintf("Test R²:   %.3f\n", cor(lm_pred, test$Performance_Rating)^2))
cat(sprintf("Test MAE:  %.2f\n", mean(abs(lm_pred - test$Performance_Rating))))
cat(sprintf("Test RMSE: %.2f\n", sqrt(mean((lm_pred - test$Performance_Rating)^2))))

# --- Logistic regression ----------------------------------------------------
glm_fit <- glm(High_Performer ~ Risk_Adjusted_Rate + Adverse_Rate +
                 Hospital_Type + Region,
               data = train, family = binomial(link = "logit"))
cat("\n--- Logistic regression summary ---\n")
print(summary(glm_fit))

probs <- predict(glm_fit, newdata = test, type = "response")
preds <- as.integer(probs >= 0.5)

cat(sprintf("\nTest accuracy: %.3f\n",
            mean(preds == test$High_Performer)))
cat(sprintf("Test AUC:      %.3f\n",
            as.numeric(auc(test$High_Performer, probs))))

cat("\nOdds ratios:\n")
print(round(exp(coef(glm_fit)), 3))
