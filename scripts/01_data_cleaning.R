############################################################
# COS80023 – D/HD Project
# 01_data_cleaning.R
#
# Purpose:
#   - Load raw Victorian crash data from /data
#   - Perform basic ETL-style cleaning (Extract–Transform–Load)
#   - Prepare a merged, cleaned dataset ready for:
#       * feature engineering
#       * K-means clustering
#       * predictive modelling
#
############################################################

## 0. Setup -------------------------------------------------

# Load packages used throughout the project
library(tidyverse)
library(janitor)     # for clean_names()
library(lubridate)   # for date/time handling

# Check working directory
getwd()

# Ensure output folders exist
if (!dir.exists("output")) dir.create("output")
if (!dir.exists("output/processed")) dir.create("output/processed")


## 1. Load raw data -----------------------------------------
accident_raw <- read_csv("datasets/accident.csv")
node_raw     <- read_csv("datasets/node.csv")
atmo_raw     <- read_csv("datasets/atmospheric_cond.csv")
road_raw     <- read_csv("datasets/road_surface_cond.csv")

# Quick structure checks
glimpse(accident_raw)
glimpse(node_raw)
glimpse(atmo_raw)
glimpse(road_raw)


## 2. Standardise column names ------------------------------

# Use janitor::clean_names() to get lower_snake_case
accident <- accident_raw  %>% clean_names()
node     <- node_raw      %>% clean_names()
atmo     <- atmo_raw      %>% clean_names()
road     <- road_raw      %>% clean_names()

# View names to decide what to keep
names(accident)
names(node)
names(atmo)
names(road)


## 3. Select relevant variables -----------------------------

# accident table
accident_sel <- accident %>%
  select(
    accident_no,          # unique crash ID
    accident_date,        # date of crash
    accident_time,        # time of crash
    day_of_week,          # existing day-of-week field
    light_condition,
    road_geometry,
    speed_zone,
    dca_desc,
    no_persons_killed,
    no_persons_inj_2,
    severity,
    node_id               # kept in case we needed it later
  )

# node table
node_sel <- node %>%
  select(
    accident_no,
    node_id,
    node_type,
    latitude,
    longitude,
    lga_name,
    deg_urban_name,
    postcode_crash
  )

# atmospheric conditions
atmo_sel <- atmo %>%
  filter(atmosph_cond_seq == 1) %>%
  select(
    accident_no,
    weather = atmosph_cond_desc
  )

# road surface conditions
road_sel <- road %>%
  filter(surface_cond_seq == 1) %>%
  select(
    accident_no,
    road_surface = surface_cond_desc
  )


## 4. Handle missing values & basic filtering ---------------

# Remove duplicate accident rows
accident_sel <- accident_sel %>%
  distinct()


## 5. Convert date & time to proper types -------------------

accident_sel <- accident_sel %>%
  mutate(
    # Ensure accident_date is a proper Date
    accident_date = as_date(accident_date),
    
    # Extract hour directly from time object
    accident_hour = hour(accident_time),
    
    # Optional: extract minutes
    accident_min = minute(accident_time),
    
    # Re-derive day_of_week and weekend flag 
    day_of_week = wday(accident_date, label = TRUE),
    is_weekend  = day_of_week %in% c("Sat", "Sun")
  )

# Keep only the useful time fields plus original context columns
accident_sel <- accident_sel %>%
  select(
    accident_no,
    accident_date,
    accident_hour,
    day_of_week,
    is_weekend,
    light_condition,
    road_geometry,
    speed_zone,
    dca_desc,
    no_persons_killed,
    no_persons_inj_2,
    severity,
    node_id
  )


## 6. Merge tables -----------------

# All four tables share accident_no, thus join on that.
acc_node <- accident_sel %>%
  left_join(node_sel, by = "accident_no")

acc_node_atmo <- acc_node %>%
  left_join(atmo_sel, by = "accident_no")

crash_full <- acc_node_atmo %>%
  left_join(road_sel, by = "accident_no")

crash_full <- crash_full %>%
  select(-node_id.y) %>%
  rename(node_id = node_id.x)

# Quick sanity check
glimpse(crash_full)


## 7. Filter out unusable rows ------------------------------

crash_clean <- crash_full %>%
  filter(
    !is.na(latitude),
    !is.na(longitude),
    !is.na(accident_hour)
  ) %>%
  # Drop impossible coordinates
  filter(
    latitude  > -90, latitude  < 90,
    longitude > -180, longitude < 180
  ) %>%
  # Filter for coordinates ONLY in Victoria
  filter(
    between(latitude, -39.2, -33.9),
    between(longitude, 140.9, 150.1)
  )

# Check missingness summary
crash_clean %>%
  summarise(across(everything(), ~ sum(is.na(.))))


## 8. Save cleaned dataset ----------------------------------

write_rds(crash_clean, "output/processed/crash_clean.rds")
write_csv(crash_clean, "output/processed/crash_clean.csv")