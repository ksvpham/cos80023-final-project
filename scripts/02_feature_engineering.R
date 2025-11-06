############################################################
# COS80023 – D/HD Project
# 02_feature_engineering.R
############################################################

## 0. Setup -------------------------------------------------
library(tidyverse)
library(lubridate)
library(stringr)
library(forcats)

if (!dir.exists("output")) dir.create("output")
if (!dir.exists("output/processed")) dir.create("output/processed")

## Helpers --------------------------------------------------
recode_road_surface <- function(x) {
  x0 <- x %>% str_trim() %>% str_squish() %>% str_to_title()
  lab <- dplyr::case_when(
    str_detect(x0, "^Dry")                          ~ "Dry",
    str_detect(x0, "^Wet")                          ~ "Wet",
    str_detect(x0, regex("unk",  ignore_case=TRUE)) ~ "Unk.",
    str_detect(x0, regex("ice",  ignore_case=TRUE)) ~ "Icy",
    str_detect(x0, regex("mud",  ignore_case=TRUE)) ~ "Muddy",
    str_detect(x0, regex("snow", ignore_case=TRUE)) ~ "Snowy",
    TRUE ~ x0
  )
  factor(lab, levels = c("Dry","Wet","Unk.","Icy","Muddy","Snowy"))
}

recode_weather <- function(x) {
  x0 <- x %>% str_trim() %>% str_squish() %>% str_to_lower()
  lab <- dplyr::case_when(
    str_detect(x0, "^(clear|fine|sunny)")         ~ "Clear",
    str_detect(x0, "^(unknown|not known|n/?a)$")  ~ "Not_known",
    str_detect(x0, "(rain|shower|drizzle|storm)") ~ "Raining",
    str_detect(x0, "(strong wind|gale|windy)")    ~ "Strong_winds",
    str_detect(x0, "(fog|mist)")                  ~ "Fog",
    str_detect(x0, "dust")                        ~ "Dust",
    str_detect(x0, "(smoke|smog)")                ~ "Smoke",
    str_detect(x0, "(snow|blizzard)")             ~ "Snowing",
    TRUE                                          ~ "Not_known"
  )
  factor(lab, levels = c("Clear","Not_known","Raining","Strong_winds",
                         "Fog","Dust","Smoke","Snowing"))
}

mk_dummies <- function(f, levels, prefix) {
  f_chr <- as.character(f)
  tibble::as_tibble(
    setNames(
      lapply(levels, function(lvl) as.integer(f_chr == lvl)),
      paste0(prefix, levels)
    )
  )
}

## 1. Load cleaned data ------------------------------------
crash_clean <- readr::read_rds("output/processed/crash_clean.rds")
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
                         levels = c("Night","Morning","Afternoon","Evening")),
    day_of_week = factor(day_of_week, ordered = FALSE),
    is_weekend  = factor(is_weekend, levels = c(FALSE, TRUE),
                         labels = c("Weekday","Weekend")),
    season      = factor(season,
                         levels = c("Summer","Autumn","Winter","Spring"))
  )

## 3. Feature engineering – environment & context -----------
crash_features <- crash_features %>%
  mutate(
    weather      = recode_weather(weather),
    road_surface = recode_road_surface(road_surface),
    severity     = factor(severity)
  )

# Context columns that exist in crash_clean
context_cols <- c("speed_zone","node_type","lga_name","deg_urban_name","postcode_crash")
context_cols <- context_cols[context_cols %in% names(crash_features)]

crash_features <- crash_features %>%
  mutate(across(all_of(context_cols), ~ if (is.character(.x)) factor(.x) else .x))

## 4. Select variables for clustering -----------------------
cluster_base <- crash_features %>%
  select(
    accident_no, latitude, longitude, accident_hour,
    time_of_day, weather, road_surface, deg_urban_name
  ) %>%
  drop_na(latitude, longitude, accident_hour, time_of_day, weather, road_surface)

# Split factors/numeric
cluster_factors <- cluster_base %>% select(time_of_day, weather, road_surface)
cluster_numeric  <- cluster_base %>% select(accident_no, latitude, longitude, accident_hour)

# --- MANUAL dummies to GUARANTEE required columns ----------
time_lvls    <- c("Night","Morning","Afternoon","Evening")
weather_lvls <- c("Clear","Not_known","Raining","Strong_winds","Fog","Dust","Smoke","Snowing")
road_lvls    <- c("Dry","Wet","Unk.","Icy","Muddy","Snowy")

# Ensure factors have the locked levels first
cluster_factors <- cluster_factors %>%
  mutate(
    time_of_day  = factor(time_of_day,  levels = time_lvls),
    weather      = factor(weather,      levels = weather_lvls),
    road_surface = factor(road_surface, levels = road_lvls)
  )

dummy_time    <- mk_dummies(cluster_factors$time_of_day,  time_lvls,    "time_of_day")
dummy_weather <- mk_dummies(cluster_factors$weather,      weather_lvls, "weather")
dummy_road    <- mk_dummies(cluster_factors$road_surface, road_lvls,    "road_surface")

# --- Deg_urban_name dummies (handles any level names safely) ---
safe_name <- function(x) {
  x %>%
    stringr::str_trim() %>% stringr::str_squish() %>% stringr::str_to_title() %>%
    stringr::str_replace_all("[^A-Za-z0-9]+", "_") %>%
    stringr::str_replace_all("^_+|_+$", "")
}

deg_vals_raw   <- as.character(cluster_base$deg_urban_name)
deg_vals_clean <- stringr::str_to_title(stringr::str_squish(deg_vals_raw))
deg_lvls       <- sort(unique(deg_vals_clean[!is.na(deg_vals_clean)]))
deg_lvls_safe  <- safe_name(deg_lvls)

dummy_deg_list <- lapply(seq_along(deg_lvls), function(i) {
  as.integer(deg_vals_clean == deg_lvls[i])
})
names(dummy_deg_list) <- paste0("deg_urban_", deg_lvls_safe)

dummy_deg <- tibble::as_tibble(dummy_deg_list)

dummy_mat <- bind_cols(dummy_time, dummy_weather, dummy_road, dummy_deg)

# Bind with numeric columns
cluster_for_scale <- bind_cols(cluster_numeric, dummy_mat)

## 5. Normalise features for K-means ------------------------
cluster_scaled <- cluster_for_scale %>%
  mutate(across(.cols = -accident_no, .fns = ~ as.numeric(scale(.x))))

## 6. Prepare dataset for predictive modelling --------------
model_input <- crash_features %>%
  select(
    accident_no, severity, latitude, longitude, accident_hour,
    time_of_day, day_of_week, is_weekend, month, season,
    weather, road_surface, all_of(context_cols)
  ) %>%
  drop_na(severity)

## 7. Save processed outputs --------------------------------
write_rds(cluster_scaled, "output/processed/cluster_scaled.rds")
write_csv(cluster_scaled, "output/processed/cluster_scaled.csv")

write_rds(model_input, "output/processed/model_input.rds")
write_csv(model_input, "output/processed/model_input.csv")


