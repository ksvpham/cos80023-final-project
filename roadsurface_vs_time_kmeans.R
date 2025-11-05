library(tidyr)
library(dplyr)
library(lubridate)
library(ggplot2)

# Read datasets
road_surface <- read.csv("datasets/road_surface_cond.csv")
accident <- read.csv("datasets/accident.csv")

# FIll in the columns
road_surface$value <- 1

# Pivot rows (SURFACE_COND_DESC) into columns (one-hot)
road_wide <- road_surface %>%
  pivot_wider(
    names_from  = SURFACE_COND_DESC,
    values_from = value,
    values_fill = 0
  )

# Bring in ACCIDENT_TIME column
road_wide <- road_wide %>%
  left_join(accident %>% select(ACCIDENT_NO, ACCIDENT_TIME),
            by = "ACCIDENT_NO")

# Convert time to numeric hour
road_wide <- road_wide %>%
  mutate(
    ACCIDENT_TIME = lubridate::hms(ACCIDENT_TIME),
    ACCIDENT_TIME = hour(ACCIDENT_TIME) + minute(ACCIDENT_TIME) / 60
  )

# Remove unneeded ID/sequence columns (adjust names if different in your file)
road_wide <- road_wide %>%
  select(-c(SURFACE_COND, SURFACE_COND_SEQ))

# Collapse duplicates by time; one row per time with summed one-hots
surface_cols <- setdiff(names(road_wide), c("ACCIDENT_NO", "ACCIDENT_TIME"))

road_grouped <- road_wide %>%
  select(-ACCIDENT_NO) %>%
  group_by(ACCIDENT_TIME) %>%
  summarise(across(all_of(surface_cols), ~ sum(.x, na.rm = TRUE)), .groups = "drop") %>%
  arrange(ACCIDENT_TIME)

head(road_grouped)

# ---- Elbow Plot (K-means on Road Surface + Time) ----

set.seed(42)

# Build numeric feature matrix (drop zero-variance columns to prevent NaN errors)
X <- road_grouped %>%
  select(where(is.numeric)) %>%
  select(where(~ sd(.x, na.rm = TRUE) > 0)) %>%  # keep only columns with variation
  scale()

# Replace any remaining non-finite values with 0 (safety)
X[!is.finite(X)] <- 0

# Compute WSS for k = 1..10
k_max <- 10
wss <- sapply(1:k_max, function(k) {
  kmeans(X, centers = k, nstart = 25)$tot.withinss
})

# Plot Elbow
plot(1:k_max, wss, type = "b", pch = 19,
     xlab = "Number of clusters (k)",
     ylab = "Total within-cluster sum of squares",
     main = "Elbow Plot (K-means on Road Surface + Time)")


# ---- K-means on Road Surface + Time (k = 2), robust version ----
set.seed(42)

# 1) Work on rows with valid time only
rg_total <- road_grouped %>% 
  filter(!is.na(ACCIDENT_TIME))

# 2) Build numeric feature matrix and drop zero-variance columns
X_rs <- rg_total %>%
  select(where(is.numeric)) %>%
  select(where(~ sd(.x, na.rm = TRUE) > 0)) %>%  # avoid sd=0 -> NaN after scale()
  scale()

# 3) Final guard: ensure no NA/NaN/Inf remain
if (any(!is.finite(X_rs))) {
  stop("Non-finite values remain in feature matrix after filtering/scaling.")
}

# 4) Fit K-means
kmeans_rs <- kmeans(X_rs, centers = 2, nstart = 25)

# 5) Attach clusters back to the filtered data
rg_total$cluster <- kmeans_rs$cluster

# 6) Total accidents per time (sum of all one-hots)
rg_total$total_accidents <- rowSums(
  rg_total %>% select(-ACCIDENT_TIME, -cluster),
  na.rm = TRUE
)

# 7) Plot (same style as your weather plot)
ggplot(rg_total, aes(x = ACCIDENT_TIME, y = total_accidents,
               color = as.factor(cluster))) +
  geom_point(size = 2) +
  geom_line(aes(group = cluster), alpha = 0.4) +
  scale_x_continuous(breaks = seq(0, 24, 2)) +
  theme_minimal() +
  labs(
    title = "Accident Frequency by Time of Day (Road Surface + Time Clusters)",
    x = "Time of Day (Hour)",
    y = "Number of Accidents",
    color = "Cluster"
  )

# 8) Summaries (same structure as weather)
cluster_summary_rs <- rg_total %>%
  group_by(cluster) %>%
  summarise(
    total_points  = n(),
    avg_time      = mean(ACCIDENT_TIME, na.rm = TRUE),
    avg_accidents = mean(total_accidents, na.rm = TRUE),
    max_accidents = max(total_accidents, na.rm = TRUE),
    min_accidents = min(total_accidents, na.rm = TRUE),
    .groups = "drop"
  )

rs_cols <- setdiff(names(rg_total), c("ACCIDENT_TIME", "cluster", "total_accidents"))
surface_summary_rs <- rg_total %>%
  group_by(cluster) %>%
  summarise(across(all_of(rs_cols), ~ sum(.x, na.rm = TRUE)), .groups = "drop")

summary_combined_rs <- cluster_summary_rs %>% left_join(surface_summary_rs, by = "cluster")

cluster_summary_rs
surface_summary_rs
summary_combined_rs