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
    node_id               # kept in case we need it later
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
  filter(atmosph_cond_seq == 1) %>%       # main condition only
  select(
    accident_no,
    weather = atmosph_cond_desc
  )

# road surface conditions
road_sel <- road %>%
  filter(surface_cond_seq == 1) %>%       # main condition only
  select(
    accident_no,
    road_surface = surface_cond_desc
  )


## 4. Handle missing values & basic de-duplication ----------

# Remove duplicate accident rows (exact duplicates)
accident_sel <- accident_sel %>%
  distinct()

# Optional: remove exact duplicates in node table as well
node_sel <- node_sel %>%
  distinct()


## 5. Convert date & time to proper types -------------------

accident_sel <- accident_sel %>%
  mutate(
    # Ensure accident_date is a proper Date
    accident_date = as_date(accident_date),
    
    # NOTE: if accident_time is imported as character in your setup,
    # you may need: accident_time = hms::as_hms(accident_time)
    # before calling hour()/minute().
    
    # Extract hour directly from time object
    accident_hour = hour(accident_time),
    
    # Optional: extract minutes (not used later, but kept if needed)
    accident_min = minute(accident_time),
    
    # Re-derive day_of_week and weekend flag (ensures consistency)
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


## 6. Merge tables ------------------------------------------

# All four tables share accident_no, thus join on that.
acc_node <- accident_sel %>%
  left_join(node_sel, by = "accident_no")

acc_node_atmo <- acc_node %>%
  left_join(atmo_sel, by = "accident_no")

crash_full <- acc_node_atmo %>%
  left_join(road_sel, by = "accident_no")

# We now have node_id from both accident and node tables; keep the node one
crash_full <- crash_full %>%
  select(-node_id.y) %>%
  rename(node_id = node_id.x)

# NEW: basic sanity checks on merged data
n_full <- nrow(crash_full)
message("Rows after joins (crash_full): ", n_full)
glimpse(crash_full)

crash_full %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  print()


## 7. Filter out unusable / low-information rows ------------

crash_clean <- crash_full %>%
  # 1) Basic required fields must not be missing
  filter(
    !is.na(latitude),
    !is.na(longitude),
    !is.na(accident_hour),
    !is.na(severity)
  ) %>%
  
  # 2) Drop impossible coordinates
  filter(
    latitude  > -90,  latitude  < 90,
    longitude > -180, longitude < 180
  ) %>%
  
  # 3) Keep only crashes within Victoria’s bounding box
  filter(
    between(latitude, -39.2, -33.9),
    between(longitude, 140.9, 150.1)
  ) %>%
  
  # 4) Mark placeholder / unknown values as NA
  mutate(
    # Ensure key coded variables are numeric
    speed_zone      = as.integer(speed_zone),
    light_condition = as.integer(light_condition),
    road_geometry   = as.integer(road_geometry),
    severity        = as.integer(severity),
    
    # Speed zone placeholder codes (VicRoads metadata)
    speed_zone = na_if(speed_zone, 999L),  # Not known
    speed_zone = na_if(speed_zone, 777L),  # Other speed limit
    speed_zone = na_if(speed_zone, 888L),  # Camping / off-road
    speed_zone = na_if(speed_zone, 000L),  # Missing / invalid
    
    # Weather / surface "unknown" categories
    weather      = na_if(weather, "Not known"),
    weather      = na_if(weather, "Unknown"),
    road_surface = na_if(road_surface, "Unk."),
    road_surface = na_if(road_surface, "Unknown"),
    
    # In VicRoads coding, 9 is usually "Unknown" for conditions
    light_condition = if_else(light_condition == 9L, NA_integer_, light_condition),
    road_geometry   = if_else(road_geometry   == 9L, NA_integer_, road_geometry)
  ) %>%
  
  # 5) Restrict to injury crashes only
  #    1 = Fatal, 2 = Serious injury, 3 = Other injury
  filter(severity %in% c(1L, 2L, 3L)) %>%
  
  # 6) Finally drop rows where core predictors are still missing
  drop_na(
    latitude,
    longitude,
    accident_hour,
    severity,
    speed_zone,
    weather,
    road_surface,
    light_condition,
    road_geometry
  )

n_clean <- nrow(crash_clean)
message("Rows before filtering: ", n_full,
        " | after filtering: ", n_clean)

# NEW: missingness summary after filtering
crash_clean %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  print()


## 8. Final column order & save cleaned dataset --------------

# NEW: lock column order to match crash_clean.csv in the ZIP
crash_clean <- crash_clean %>%
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
    node_id,
    node_type,
    latitude,
    longitude,
    lga_name,
    deg_urban_name,
    postcode_crash,
    weather,
    road_surface
  )

write_rds(crash_clean, "output/processed/crash_clean.rds")
write_csv(crash_clean, "output/processed/crash_clean.csv")

message("✅ Cleaning complete. Clean dataset written to output/processed/crash_clean.[rds/csv]")
