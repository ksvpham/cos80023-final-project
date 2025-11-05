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
#   - Prepare:
#       * a clustering-ready, scaled dataset
#       * a modelling-ready dataset (for prediction later)
#
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
    time_of_day = factor(time_of_day,
                         levels = c("Night", "Morning", "Afternoon", "Evening")),
    day_of_week = factor(day_of_week, ordered = FALSE),
    is_weekend  = factor(is_weekend, levels = c(FALSE, TRUE),
                         labels = c("Weekday", "Weekend")),
    season      = factor(season,
                         levels = c("Summer", "Autumn", "Winter", "Spring"))
  )


## 3. Feature engineering – environment & context ------------

# Use existing columns from crash_clean
crash_features <- crash_features %>%
  mutate(
    weather      = factor(weather),
    road_surface = factor(road_surface),
    severity     = factor(severity)   # for classification later
  )

# Context columns that exist in crash_clean
context_cols <- c("speed_zone", "node_type", "lga_name",
                  "deg_urban_name", "postcode_crash")
context_cols <- context_cols[context_cols %in% names(crash_features)]

crash_features <- crash_features %>%
  mutate(across(all_of(context_cols),
                ~ if (is.character(.x)) factor(.x) else .x))


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
  drop_na(latitude, longitude, accident_hour, time_of_day, weather, road_surface)

# Factors vs numeric split
cluster_factors <- cluster_base %>%
  select(time_of_day, weather, road_surface)

cluster_numeric <- cluster_base %>%
  select(accident_no, latitude, longitude, accident_hour)

dummy_mat <- model.matrix(~ time_of_day + weather + road_surface,
                          data = cluster_factors) %>%
  as_tibble() %>%
  select(-`(Intercept)`)

cluster_for_scale <- bind_cols(
  cluster_numeric,
  dummy_mat
)

glimpse(cluster_for_scale)


## 5. Normalise features for K-means ------------------------

# Future input for 03_kmeans_analysis.R file
cluster_scaled <- cluster_for_scale %>%
  mutate(across(
    .cols = -accident_no,
    .fns  = ~ as.numeric(scale(.x))
  ))


## 6. Prepare dataset for predictive modelling --------------

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
  drop_na(severity)

## 7. Save processed outputs --------------------------------

write_rds(cluster_scaled, "output/processed/cluster_scaled.rds")
write_csv(cluster_scaled, "output/processed/cluster_scaled.csv")

write_rds(model_input, "output/processed/model_input.rds")
write_csv(model_input, "output/processed/model_input.csv")