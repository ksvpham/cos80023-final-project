############################################################
# COS80023 – D/HD Project
# 04_predictive_model.R
#
# Purpose:
#   - Use engineered dataset from 02_feature_engineering.R
#     (model_input.rds)
#   - Predict binary crash severity:
#         severity_bin = Minor vs Serious
#   - Deal with class imbalance (up-sampling)
#   - Train and compare:
#         * Decision Tree (baseline)
#         * Random Forest (ranger backend)
#         * Gradient Boosting (GBM)
#   - Look at feature importance and where/when serious
#     crashes are most likely (risk heatmap)
############################################################


## 0. Setup -------------------------------------------------

library(tidyverse)
library(caret)
library(randomForest)   # caret still likes having this around
library(ranger)
library(gbm)
library(ggplot2)

getwd()

if (!dir.exists("output")) dir.create("output")
if (!dir.exists("output/models")) dir.create("output/models")
if (!dir.exists("output/plots")) dir.create("output/plots")


## 1. Load dataset ------------------------------------------

# This was created by 02_feature_engineering.R
model_input <- read_rds("output/processed/model_input.rds")
glimpse(model_input)


## 2. Prepare data (pick columns, binary target) ------------

# We already have:
#   - severity_bin (Minor / Serious)
#   - extra helper fields (is_night, is_peak, speed_zone_serious_rate)
# Here we:
#   * drop the ID (accident_no)
#   * drop original 3-level severity (otherwise it basically
#     gives away the answer)
#   * keep rows with no missing values

data_model <- model_input %>%
  select(-accident_no, -severity) %>%
  drop_na()

# Make sure target has the right type / ordering
data_model$severity_bin <- factor(data_model$severity_bin,
                                  levels = c("Minor", "Serious"))

table(data_model$severity_bin)


## 3. Train/test split --------------------------------------

set.seed(123)
train_index <- createDataPartition(data_model$severity_bin,
                                   p = 0.7, list = FALSE)
train_data <- data_model[train_index, ]
test_data  <- data_model[-train_index, ]

nrow(train_data); nrow(test_data)


## 4. Preprocessing: scale numeric predictors ---------------

# Only scale numeric columns – factors stay as they are
num_cols <- train_data %>%
  select(where(is.numeric)) %>%
  names()

preproc <- preProcess(train_data[, num_cols],
                      method = c("center", "scale"))

train_num <- predict(preproc, train_data[, num_cols])
test_num  <- predict(preproc,  test_data[, num_cols])

# Put the scaled numerics back together with the factor cols
train_proc <- bind_cols(
  train_num,
  train_data %>% select(-all_of(num_cols))
)

test_proc <- bind_cols(
  test_num,
  test_data %>% select(-all_of(num_cols))
)


## 5. Deal with class imbalance (up-sampling) ---------------

# Serious crashes are much rarer, so we'll up-sample the
# minority class in the training data to give the models
# a fighting chance.

set.seed(123)
train_up <- upSample(
  x = train_proc %>% select(-severity_bin),
  y = train_proc$severity_bin,
  yname = "severity_bin"
)

table(train_up$severity_bin)


## 6. Baseline Model – Decision Tree ------------------------

set.seed(123)
model_tree <- train(
  severity_bin ~ .,
  data = train_up,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 5)
)

print(model_tree)
plot(model_tree)
ggsave("output/plots/model_tree_cv.png",
       plot = last_plot(), width = 6, height = 4, dpi = 300)


## 7. Random Forest – ranger backend ------------------------

set.seed(123)

# Keep training time reasonable but still use a big chunk
rf_train <- train_up %>%
  sample_n(min(40000, nrow(train_up)))

# Tell caret we want class probabilities
ctrl_rf <- trainControl(
  method = "none",
  classProbs = TRUE
)

mtry_val <- floor(sqrt(ncol(rf_train) - 1))

model_rf <- train(
  severity_bin ~ .,
  data = rf_train,
  method = "ranger",
  trControl = ctrl_rf,
  tuneGrid = data.frame(
    mtry          = mtry_val,
    splitrule     = "gini",
    min.node.size = 5
  ),
  num.trees  = 300,
  importance = "impurity",
  verbose    = TRUE
  # NOTE: do NOT add probability = TRUE here – caret handles it
)

print(model_rf)

# Out-of-bag error (1 - this ≈ OOB accuracy)
model_rf$finalModel$prediction.error


## 8. Gradient Boosting Machine (GBM) -----------------------

set.seed(123)

gbm_train <- train_up %>%
  sample_n(min(30000, nrow(train_up)))

model_gbm <- train(
  severity_bin ~ .,
  data = gbm_train,
  method = "gbm",
  trControl = trainControl(method = "cv", number = 3),
  tuneLength = 3,
  verbose = FALSE
)

print(model_gbm)
plot(model_gbm)
ggsave("output/plots/model_gbm_tuning.png",
       plot = last_plot(), width = 6, height = 4, dpi = 300)


## 9. Evaluate models on the hold-out test set --------------

# Class predictions
pred_tree <- predict(model_tree, test_proc)
pred_rf   <- predict(model_rf,   test_proc)
pred_gbm  <- predict(model_gbm,  test_proc)

cm_tree <- confusionMatrix(pred_tree, test_proc$severity_bin)
cm_rf   <- confusionMatrix(pred_rf,   test_proc$severity_bin)
cm_gbm  <- confusionMatrix(pred_gbm,  test_proc$severity_bin)

cm_tree
cm_rf
cm_gbm

# Quick comparison table (Accuracy + Kappa)
model_results <- tibble(
  Model   = c("Decision Tree", "Random Forest (ranger)", "GBM"),
  Accuracy = c(cm_tree$overall["Accuracy"],
               cm_rf$overall["Accuracy"],
               cm_gbm$overall["Accuracy"]),
  Kappa    = c(cm_tree$overall["Kappa"],
               cm_rf$overall["Kappa"],
               cm_gbm$overall["Kappa"])
)

print(model_results)

acc_plot <- ggplot(model_results,
                   aes(x = Model, y = Accuracy, fill = Model)) +
  geom_col() +
  geom_text(aes(label = round(Accuracy, 3)),
            vjust = -0.3) +
  ylim(0, 1) +
  labs(title = "Model Accuracy Comparison\n(Minor vs Serious crashes)",
       y = "Accuracy") +
  theme_minimal()

print(acc_plot)

ggsave("output/plots/model_accuracy_comparison.png",
       plot = acc_plot, width = 7, height = 4, dpi = 300)


## 10. Feature Importance -----------------------------------

# Mainly use RF importance, since it's usually the most stable
importance_rf <- varImp(model_rf)
plot(importance_rf, top = 15,
     main = "Top 15 Important Features (Random Forest)")

ggsave("output/plots/model_rf_feature_importance.png",
       plot = last_plot(), width = 7, height = 5, dpi = 300)

# Optional: GBM importance for comparison
importance_gbm <- varImp(model_gbm)
plot(importance_gbm, top = 15,
     main = "Top 15 Important Features (GBM)")

ggsave("output/plots/model_gbm_feature_importance.png",
       plot = last_plot(), width = 7, height = 5, dpi = 300)


## 11. Risk heatmap: when/where serious crashes are likely --

# Use RF predicted probabilities for "Serious" crashes
rf_probs <- predict(model_rf, newdata = test_proc, type = "prob")

# rf_probs is a data.frame with columns "Minor" and "Serious"
test_with_preds <- test_proc %>%
  mutate(serious_prob_rf = rf_probs[ , "Serious"])

# Average predicted risk by (scaled) speed zone and time of day
risk_speed_time <- test_with_preds %>%
  group_by(speed_zone, time_of_day) %>%
  summarise(
    mean_serious_prob = mean(serious_prob_rf),
    n = n(),
    .groups = "drop"
  )

risk_heatmap <- ggplot(risk_speed_time,
                       aes(x = time_of_day,
                           y = speed_zone,
                           fill = mean_serious_prob)) +
  geom_tile(colour = "grey80") +
  scale_fill_viridis_c(option = "plasma") +
  labs(
    title = "Predicted probability of SERIOUS crash\nby speed zone (scaled) and time of day",
    x = "Time of day",
    y = "Speed zone (scaled)",
    fill = "Serious crash\nprobability"
  ) +
  theme_minimal()

print(risk_heatmap)

ggsave("output/plots/risk_heatmap_speed_time.png",
       plot = risk_heatmap, width = 8, height = 5, dpi = 300)


## 12. Save models + metrics --------------------------------

saveRDS(model_tree, "output/models/model_tree_bin.rds")
saveRDS(model_rf,   "output/models/model_rf_bin.rds")
saveRDS(model_gbm,  "output/models/model_gbm_bin.rds")

write_csv(model_results,
          "output/models/model_performance_bin.csv")