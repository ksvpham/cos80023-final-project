############################################################
# COS80023 – D/HD Project
# 02_feature_engineering.R
#
# Purpose:
#   - Start from cleaned, integrated crash dataset produced
#     by 01_data_cleaning.R  (crash_clean.rds)
#   - Create new features:
#       * hour bins / time-of-day
#       * weekday vs weekend
#       * month / season
#       * encoded weather and road surface
#       * binary severity for modelling
#       * a couple of simple interaction-style features
#   - Prepare:
#       * a clustering-ready, scaled dataset
#       * a modelling-ready dataset (for prediction later)
############################################################

## 0. Setup -------------------------------------------------

library(tidyverse)
library(lubridate)

getwd()

if (!dir.exists("output")) dir.create("output")
if (!dir.exists("output/processed")) dir.create("output/processed")


## 1. Load cleaned data -------------------------------------

crash_clean <- read_rds("output/processed/crash_clean.rds")
glimpse(crash_clean)


## 2. Feature engineering – time-based ----------------------

crash_features <- crash_clean %>%
  mutate(
    # Time-of-day bins
    time_of_day = case_when(
      accident_hour >= 5  & accident_hour < 12 ~ "Morning",
      accident_hour >= 12 & accident_hour < 17 ~ "Afternoon",
      accident_hour >= 17 & accident_hour < 21 ~ "Evening",
      TRUE                                     ~ "Night"
    ),
    # Month & season
    month  = month(accident_date),
    season = case_when(
      month %in% c(12, 1, 2) ~ "Summer",
      month %in% c(3, 4, 5)  ~ "Autumn",
      month %in% c(6, 7, 8)  ~ "Winter",
      TRUE                   ~ "Spring"
    )
  ) %>%
  mutate(
    # Ordered time-of-day is nice for plots / interpretation
    time_of_day = factor(
      time_of_day,
      levels = c("Night", "Morning", "Afternoon", "Evening")
    ),
    # Keep day_of_week as an unordered factor (Mon, Tue, ...)
    day_of_week = factor(day_of_week, ordered = FALSE),
    # Make weekend explicit as factor
    is_weekend  = factor(is_weekend,
                         levels = c(FALSE, TRUE),
                         labels = c("Weekday", "Weekend")),
    season      = factor(
      season,
      levels = c("Summer", "Autumn", "Winter", "Spring")
    )
  )


## 3. Feature engineering – environment & context -----------

crash_features <- crash_features %>%
  mutate(
    weather      = factor(weather),
    road_surface = factor(road_surface),
    # EXPLICIT severity encoding: numeric 1/2/3 -> labelled factor
    severity = factor(
      severity,
      levels = c(1L, 2L, 3L),
      labels = c("Fatal", "Serious injury", "Other injury")
    )
  )

# Context columns that exist in crash_clean
context_cols <- c("speed_zone", "node_type", "lga_name",
                  "deg_urban_name", "postcode_crash")
context_cols <- context_cols[context_cols %in% names(crash_features)]

# Convert character context columns to factors; keep numeric ones as-is
crash_features <- crash_features %>%
  mutate(across(
    all_of(context_cols),
    ~ if (is.character(.x)) factor(.x) else .x
  ))

# ---- Extra features for modelling -------------------------

crash_features <- crash_features %>%
  mutate(
    # Binary target: Serious vs Minor – used later in 04_... modelling
    severity_bin = if_else(
      severity %in% c("Fatal", "Serious injury"),
      "Serious",
      "Minor"
    ),
    # Simple time flags (extra predictors)
    is_night = time_of_day %in% c("Evening", "Night"),
    is_peak  = accident_hour %in% c(7:9, 16:18)
  )

crash_features$severity_bin <- factor(
  crash_features$severity_bin,
  levels = c("Minor", "Serious")
)

# Speed-zone "prior risk" feature: proportion of serious crashes
speed_zone_risk <- crash_features %>%
  group_by(speed_zone) %>%
  summarise(
    speed_zone_serious_rate =
      mean(severity_bin == "Serious", na.rm = TRUE),
    .groups = "drop"
  )

crash_features <- crash_features %>%
  left_join(speed_zone_risk, by = "speed_zone")


## 4. Select variables for clustering -----------------------

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
  drop_na(
    latitude, longitude, accident_hour,
    time_of_day, weather, road_surface
  )

# Split factors vs numeric
cluster_factors <- cluster_base %>%
  select(time_of_day, weather, road_surface)

cluster_numeric <- cluster_base %>%
  select(accident_no, latitude, longitude, accident_hour)

# Create dummy variables for the categorical predictors
dummy_mat <- model.matrix(
  ~ time_of_day + weather + road_surface,
  data = cluster_factors
) %>%
  as_tibble() %>%
  select(-`(Intercept)`)

# Combined dataset for scaling
cluster_for_scale <- bind_cols(
  cluster_numeric,
  dummy_mat
)

glimpse(cluster_for_scale)


## 5. Normalise features for K-means ------------------------

cluster_scaled <- cluster_for_scale %>%
  mutate(across(
    .cols = -accident_no,      # keep ID unscaled
    .fns  = ~ as.numeric(scale(.x))
  ))

# Optional quick check
glimpse(cluster_scaled)


## 6. Prepare dataset for predictive modelling --------------

# model_input now includes:
#   - severity (3-level factor: Fatal / Serious injury / Other injury)
#   - severity_bin (Minor vs Serious)
#   - spatial, temporal, environment, and context features
model_input <- crash_features %>%
  select(
    accident_no,
    severity,
    severity_bin,
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
    all_of(context_cols),
    is_night,
    is_peak,
    speed_zone_serious_rate
  ) %>%
  drop_na(severity_bin)

# Optional sanity check
glimpse(model_input)


## 7. Save processed outputs --------------------------------

write_rds(cluster_scaled, "output/processed/cluster_scaled.rds")
write_csv(cluster_scaled, "output/processed/cluster_scaled.csv")

write_rds(model_input, "output/processed/model_input.rds")
write_csv(model_input, "output/processed/model_input.csv")

message("✅ Feature engineering complete: cluster_scaled & model_input written to output/processed/")
