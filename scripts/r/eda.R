# Exploratory analysis in R — mirrors scripts/python/02_eda.py
# Run from project root: Rscript scripts/r/eda.R

library(tidyverse)
library(corrplot)

df <- read_csv("data/hospital_clean.csv", show_col_types = FALSE)

cat("\n--- By Hospital Type ---\n")
df |>
  group_by(Hospital_Type) |>
  summarise(n = n(),
            mean_rating = mean(Performance_Rating),
            sd_rating = sd(Performance_Rating)) |>
  print()

cat("\n--- By Region ---\n")
df |>
  group_by(Region) |>
  summarise(n = n(),
            mean_rating = mean(Performance_Rating),
            sd_rating = sd(Performance_Rating)) |>
  print()

dir.create("figures", showWarnings = FALSE)

ggplot(df, aes(Performance_Rating)) +
  geom_histogram(bins = 20, fill = "steelblue", colour = "white") +
  labs(title = "Performance Rating distribution",
       x = "Performance Rating", y = "Count") +
  theme_minimal()
ggsave("figures/r_rating_distribution.png", width = 7, height = 4, dpi = 150)

ggplot(df, aes(Hospital_Type, Performance_Rating, fill = Hospital_Type)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Performance Rating by Hospital Type") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave("figures/r_rating_by_type.png", width = 7, height = 4, dpi = 150)

ggplot(df, aes(Risk_Adjusted_Rate, Performance_Rating)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", formula = y ~ x) +
  labs(title = "Risk-Adjusted Mortality vs Performance Rating") +
  theme_minimal()
ggsave("figures/r_rate_vs_rating.png", width = 7, height = 5, dpi = 150)

num_cols <- c("Performance_Rating", "Risk_Adjusted_Rate", "Cases",
              "Adverse_Events", "Adverse_Rate")
png("figures/r_correlation_matrix.png", width = 800, height = 700, res = 150)
corrplot(cor(df[, num_cols]), method = "color", addCoef.col = "black",
         tl.col = "black", tl.srt = 45, number.cex = 0.8)
dev.off()

cat("\nWrote figures to ./figures\n")
