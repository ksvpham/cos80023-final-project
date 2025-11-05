############################################################
# COS80023 – HD Project
# 02_feature_engineering.R
#
# Purpose:
#   - Start from cleaned, integrated crash dataset produced
#     by 01_data_cleaning.R  (crash_clean.rds)
#   - Create new features as per Project Plan:
#       * hour bins / time-of-day
#       * weekday vs weekend
#       * month / season
#       * encoded weather and road surface
#   - Prepare:
#       * a clustering-ready, scaled dataset
#       * a modelling-ready dataset (for prediction later)
#
# This script corresponds to Project Plan stages:
#   Stage 4: Feature Engineering
#   Stage 5: Data Transformation & Normalisation
############################################################


## 0. Setup -------------------------------------------------

library(tidyverse)
library(lubridate)

# Optional: check working directory
getwd()

# Ensure output folders exist
if (!dir.exists("output")) dir.create("output")
if (!dir.exists("output/processed")) dir.create("output/processed")


## 1. Load cleaned data -------------------------------------

# Input produced by 01_data_cleaning.R
crash_clean <- read_rds("output/processed/crash_clean.rds")

glimpse(crash_clean)


## 2. Feature engineering – time-based ----------------------

# According to the plan:
#   - use time of day, weekday/weekend, season
#   - accident_hour and accident_date were already created in 01 script

crash_features <- crash_clean %>%
  mutate(
    # Time-of-day bins (you can adjust thresholds if needed)
    time_of_day = case_when(
      accident_hour >= 5  & accident_hour < 12 ~ "Morning",
      accident_hour >= 12 & accident_hour < 17 ~ "Afternoon",
      accident_hour >= 17 & accident_hour < 21 ~ "Evening",
      TRUE                                     ~ "Night"
    ),
    
    # Month & season for potential seasonal effects
    month       = month(accident_date),
    season      = case_when(
      month %in% c(12, 1, 2) ~ "Summer",
      month %in% c(3, 4, 5)  ~ "Autumn",
      month %in% c(6, 7, 8)  ~ "Winter",
      TRUE                   ~ "Spring"
    )
  )

# Convert key fields to factors for interpretability & modelling
crash_features <- crash_features %>%
  mutate(
    time_of_day  = factor(time_of_day,
                          levels = c("Night", "Morning", "Afternoon", "Evening")),
    day_of_week  = factor(day_of_week, ordered = FALSE),
    is_weekend   = factor(is_weekend, levels = c(FALSE, TRUE),
                          labels = c("Weekday", "Weekend")),
    season       = factor(season,
                          levels = c("Summer", "Autumn", "Winter", "Spring"))
  )


## 3. Feature engineering – environment & context ------------

# In line with the plan: include weather and road surface
# Actual column names may vary; adjust if needed.
# We assume 01_data_cleaning kept:
#   - atmospheric_cond
#   - road_surface_cond
#   - severity (for later prediction)
#   - plus node-level features like speed_zone or road_type if present.

crash_features <- crash_features %>%
  mutate(
    weather      = factor(atmospheric_cond),
    road_surface = factor(road_surface_cond),
    severity     = factor(severity)   # keep as factor for classification
  )

# Optional: Node / road context – only if these columns exist
context_cols <- c("speed_zone", "road_type", "region_name", "urban_rural", "node_type")
context_cols <- context_cols[context_cols %in% names(crash_features)]

# For any of these that are character, convert to factor
crash_features <- crash_features %>%
  mutate(across(all_of(context_cols),
                ~ if (is.character(.x)) factor(.x) else .x))


## 4. Select variables for clustering -----------------------

# Project Plan for K-means:
#   "Apply K-means to group accidents by time, weather and coordinates"
#   -> core features:
#        - latitude, longitude (location)
#        - accident_hour / time_of_day
#        - weather (encoded)
#
# For K-means we need numeric variables only.
# We'll:
#   1) Keep a small, focused set of variables
#   2) Dummy-encode categorical variables
#   3) Scale all numeric features.

cluster_base <- crash_features %>%
  select(
    accident_no,
    latitude,
    longitude,
    accident_hour,
    time_of_day,
    weather,
    road_surface
  ) %>%
  drop_na(latitude, longitude, accident_hour, time_of_day, weather)

# Create dummy variables for time_of_day, weather, road_surface
# using model.matrix (base R pattern, similar to course style)

cluster_factors <- cluster_base %>%
  select(time_of_day, weather, road_surface)

cluster_numeric <- cluster_base %>%
  select(accident_no, latitude, longitude, accident_hour)

# model.matrix creates an intercept by default; we remove it.
dummy_mat <- model.matrix(~ time_of_day + weather + road_surface,
                          data = cluster_factors) %>%
  as_tibble() %>%
  select(-`(Intercept)`)

# Combine numeric and dummy features
cluster_for_scale <- bind_cols(
  cluster_numeric,
  dummy_mat
)

# Check structure
glimpse(cluster_for_scale)


## 5. Normalise features for K-means ------------------------

# In line with the plan:
#   "standardise numeric variables (as in weekly clustering practicals)"
# We scale all feature columns except accident_no (identifier).

cluster_scaled <- cluster_for_scale %>%
  mutate(
    across(
      .cols = -accident_no,
      .fns  = scale
    )
  )

# This object (cluster_scaled) will be the main input to:
#   03_kmeans_analysis.R


## 6. Prepare dataset for predictive modelling --------------

# For predictive models (later, 04_predictive_model.R),
# we keep:
#   - severity as the target
#   - engineered time features
#   - weather / road_surface
#   - optional context like speed_zone, road_type, etc.
# Scaling and dummy encoding for models can be done inside
# the modelling script using caret, to follow the ML lecture style.

model_input <- crash_features %>%
  select(
    accident_no,
    severity,
    latitude,
    longitude,
    accident_hour,
    time_of_day,
    day_of_week,
    is_weekend,
    month,
    season,
    weather,
    road_surface,
    all_of(context_cols)
  ) %>%
  drop_na(severity)   # ensure we have labels for supervised learning


## 7. Save processed outputs --------------------------------

# Clustering input (numeric, scaled)
write_rds(cluster_scaled, "output/processed/cluster_scaled.rds")
write_csv(cluster_scaled, "output/processed/cluster_scaled.csv")

# Modelling input (engineered features, not yet scaled/dummified)
write_rds(model_input, "output/processed/model_input.rds")
write_csv(model_input, "output/processed/model_input.csv")

############################################################
# End of 02_feature_engineering.R
#
# Next scripts in the plan:
#   - 03_kmeans_analysis.R   (Stage 6: K-means Clustering)
#   - 04_predictive_model.R  (Stage 8: Predictive Modelling)
############################################################
