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
library(stringr)
library(forcats)

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

recode_road_surface <- function(x) {
  x <- x %>% str_trim() %>% str_squish() %>% str_to_title()
  x <- dplyr::case_when(
    str_detect(x, "^Dry") ~ "Dry",
    str_detect(x, "^Wet") ~ "Wet",
    str_detect(x, regex("unk",  ignore_case = TRUE)) ~ "Unk.",
    str_detect(x, regex("ice",  ignore_case = TRUE)) ~ "Icy",
    str_detect(x, regex("mud",  ignore_case = TRUE)) ~ "Muddy",
    str_detect(x, regex("snow", ignore_case = TRUE)) ~ "Snowy",
    TRUE ~ x
  )
  factor(x, levels = c("Dry","Wet","Unk.","Icy","Muddy","Snowy"))  # lock levels
}

crash_features <- crash_features %>%
  mutate(
    weather      = factor(weather),
    road_surface = recode_road_surface(road_surface),
    severity     = factor(severity)
  )

# Context columns (unchanged)
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
cluster_factors <- cluster_base %>% select(time_of_day, weather, road_surface)
cluster_numeric  <- cluster_base %>% select(accident_no, latitude, longitude, accident_hour)

# One-hot encode ALL levels (no intercept)
dummy_mat <- model.matrix(~ . + 0, data = cluster_factors) %>% as_tibble()

# --- Ensure a clean road_surfaceDry column exists -------------------------
# 1) Normalise any dots in names (e.g., "Unk." often becomes "Unk.")
names(dummy_mat) <- gsub("\\.+$", "", names(dummy_mat))   # drop trailing dots
names(dummy_mat) <- gsub("\\.+", "_", names(dummy_mat))   # internal dots -> _

# 2) If model.matrix didn’t emit road_surfaceDry, create it explicitly
if (!"road_surfaceDry" %in% names(dummy_mat)) {
  # Create it from the original factor
  dummy_mat$road_surfaceDry <- as.integer(as.character(cluster_factors$road_surface) == "Dry")
}

# (Optional) Ensure the rest of road_surface dummies exist too
for (lev in c("Wet","Unk.","Icy","Muddy","Snowy")) {
  col <- paste0("road_surface", gsub("\\.", "", lev))  # "Unk." -> "Unk"
  if (!col %in% names(dummy_mat)) {
    dummy_mat[[col]] <- as.integer(as.character(cluster_factors$road_surface) == lev)
  }
}

# Bind with numeric columns
cluster_for_scale <- bind_cols(cluster_numeric, dummy_mat)

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