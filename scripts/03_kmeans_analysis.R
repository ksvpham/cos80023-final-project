############################################################
# COS80023 – D/HD Project
# 03_kmeans_analysis.R
#
# Purpose:
#   - Load scaled clustering dataset from 02_feature_engineering.R
#   - Use elbow method to choose k (number of clusters)
#   - Fit K-means model on:
#       * latitude, longitude
#       * time-of-day & weather & road surface dummies
#   - Attach cluster labels back to crashes for mapping
#   - Visualise clusters on a map of Victoria and interpret
#
############################################################


## 0. Setup -------------------------------------------------

library(tidyverse)
library(sf)
library(ozmaps)

getwd()

if (!dir.exists("output")) dir.create("output")
if (!dir.exists("output/processed")) dir.create("output/processed")
if (!dir.exists("output/plots")) dir.create("output/plots")


## 1. Load clustering data ----------------------------------

# Input produced by 02_feature_engineering.R
cluster_scaled <- read_rds("output/processed/cluster_scaled.rds")

glimpse(cluster_scaled)
nrow(cluster_scaled)


## 2. Elbow method to choose k ------------------------------

# Remove ID column for K-means, keep only numeric features
cluster_mat <- cluster_scaled %>%
  select(-accident_no) %>%
  as.matrix()

# Function to compute total within-cluster sum of squares for a given k
compute_wss <- function(k) {
  kmeans(cluster_mat, centers = k, nstart = 10)$tot.withinss
}

# Range of k values to test
k_values <- 2:10

wss_values <- tibble(
  k   = k_values,
  wss = map_dbl(k_values, compute_wss)
)

# Plot elbow curve
elbow_plot <- ggplot(wss_values, aes(x = k, y = wss)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Elbow Method for K-means",
    x = "Number of clusters (k)",
    y = "Total within-cluster sum of squares"
  )

print(elbow_plot)

ggsave("output/plots/kmeans_elbow_plot.png",
       plot = elbow_plot, width = 6, height = 4, dpi = 300)


## 3. Fit final K-means model -------------------------------

# Choose k based on the elbow plot.
optimal_k <- 6   # NOTE: change this if elbow suggests a different k

set.seed(123)    # for reproducibility

kmeans_fit <- kmeans(
  x       = cluster_mat,
  centers = optimal_k,
  nstart  = 25
)

# Add cluster labels back to cluster_scaled
cluster_with_labels <- cluster_scaled %>%
  mutate(
    cluster = factor(kmeans_fit$cluster)
  )

# Quick sanity check
cluster_with_labels %>%
  count(cluster)


## 4. Attach clusters to full crash data --------------------

# Load the richer feature set for interpretation / mapping
model_input <- read_rds("output/processed/model_input.rds")

# IMPORTANT:
# Because model_input and cluster_scaled both come from the same
# crash_features object (and we never re-ordered rows), we can
# safely align by row position instead of joining by accident_no.
stopifnot(
  nrow(model_input) == nrow(cluster_with_labels),
  identical(model_input$accident_no, cluster_with_labels$accident_no)
)

crash_with_clusters <- model_input %>%
  mutate(cluster = cluster_with_labels$cluster)

# Check structure
glimpse(crash_with_clusters)

# Quick summary: clusters by severity, season, time-of-day, etc.
crash_with_clusters %>%
  count(cluster, severity)

crash_with_clusters %>%
  count(cluster, time_of_day)

crash_with_clusters %>%
  count(cluster, season)


## 5. Basic spatial view (scatter by cluster) ---------------

cluster_scatter <- ggplot(crash_with_clusters,
                          aes(x = longitude, y = latitude, colour = cluster)) +
  geom_point(alpha = 0.3, size = 0.5) +
  coord_fixed() +
  labs(
    title = "Crash clusters across Victoria (K-means)",
    x = "Longitude",
    y = "Latitude",
    colour = "Cluster"
  )

print(cluster_scatter)

ggsave("output/plots/kmeans_clusters_scatter.png",
       plot = cluster_scatter, width = 6, height = 5, dpi = 300)


## 6. Map clusters on a Victoria boundary -------------------

# Get Victoria polygon from ozmaps
vic <- ozmaps::ozmap_states %>%
  filter(NAME == "Victoria")

# K-means clusters laid over actual Victoria map
vic_cluster_map <- ggplot() +
  geom_sf(data = vic, fill = "grey95", colour = "grey70") +
  geom_point(
    data  = crash_with_clusters,
    aes(x = longitude, y = latitude, colour = cluster),
    alpha = 0.3,
    size  = 0.4
  ) +
  coord_sf(
    xlim = c(140.9, 150.1),
    ylim = c(-39.2, -33.9),
    expand = FALSE
  ) +
  labs(
    title = "Crash clusters across Victoria (K-means on map)",
    x = "Longitude",
    y = "Latitude",
    colour = "Cluster"
  ) +
  theme_minimal()

print(vic_cluster_map)

ggsave("output/plots/kmeans_clusters_victoria_map.png",
       plot = vic_cluster_map, width = 7, height = 6, dpi = 300)


## 7. Faceted mini-maps by cluster --------------------------

faceted_map <- ggplot(crash_with_clusters,
                      aes(x = longitude, y = latitude)) +
  geom_point(alpha = 0.3, size = 0.3, colour = "steelblue") +
  coord_fixed() +
  facet_wrap(~ cluster) +
  labs(
    title = "Crash locations by K-means cluster",
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_minimal()

print(faceted_map)

ggsave("output/plots/kmeans_clusters_faceted_map.png",
       plot = faceted_map, width = 8, height = 6, dpi = 300)


## 8. Time-of-day distribution by cluster -------------------

tod_by_cluster <- crash_with_clusters %>%
  count(cluster, time_of_day) %>%
  group_by(cluster) %>%
  mutate(prop = n / sum(n))

tod_plot <- ggplot(tod_by_cluster,
                   aes(x = time_of_day, y = prop, fill = time_of_day)) +
  geom_col() +
  facet_wrap(~ cluster) +
  labs(
    title = "Time-of-day distribution by cluster",
    x = "Time of day",
    y = "Proportion of crashes"
  ) +
  theme_minimal()

print(tod_plot)

ggsave("output/plots/kmeans_time_of_day_by_cluster.png",
       plot = tod_plot, width = 8, height = 6, dpi = 300)


## 9. Severity distribution by cluster ----------------------

severity_by_cluster <- crash_with_clusters %>%
  count(cluster, severity) %>%
  group_by(cluster) %>%
  mutate(prop = n / sum(n))

severity_plot <- ggplot(severity_by_cluster,
                        aes(x = cluster, y = prop, fill = severity)) +
  geom_col(position = "fill") +
  labs(
    title = "Severity distribution by cluster",
    x = "Cluster",
    y = "Proportion of crashes",
    fill = "Severity"
  ) +
  theme_minimal()

print(severity_plot)

ggsave("output/plots/kmeans_severity_by_cluster.png",
       plot = severity_plot, width = 7, height = 5, dpi = 300)


## 10. Crash density heatmap over Victoria ------------------

density_plot <- ggplot(crash_with_clusters,
                       aes(x = longitude, y = latitude)) +
  geom_density_2d_filled(contour_var = "ndensity", alpha = 0.8) +
  facet_wrap(~ cluster) +
  coord_sf(
    xlim = c(140.9, 150.1),
    ylim = c(-39.2, -33.9),
    expand = FALSE
  ) +
  labs(
    title = "Crash density by cluster across Victoria",
    x = "Longitude",
    y = "Latitude",
    fill = "Relative density"
  ) +
  theme_minimal()

print(density_plot)

ggsave("output/plots/kmeans_crash_density_by_cluster.png",
       plot = density_plot, width = 8, height = 6, dpi = 300)


## 11. Weather distribution by cluster ----------------------

weather_by_cluster <- crash_with_clusters %>%
  count(cluster, weather) %>%            # count crashes by (cluster, weather)
  group_by(cluster) %>%
  mutate(prop = n / sum(n))             # turn counts into proportions

weather_plot <- ggplot(weather_by_cluster,
                       aes(x = weather, y = prop, fill = weather)) +
  geom_col() +
  facet_wrap(~ cluster) +
  labs(
    title = "Weather conditions by cluster",
    x = "Weather condition",
    y = "Proportion of crashes"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"            # optional: colours are shown on x-axis already
  )

print(weather_plot)

ggsave("output/plots/kmeans_weather_by_cluster.png",
       plot = weather_plot, width = 8, height = 6, dpi = 300)


## 12. Save clustering outputs ------------------------------

write_rds(cluster_with_labels,
          "output/processed/cluster_with_labels.rds")
write_csv(cluster_with_labels,
          "output/processed/cluster_with_labels.csv")

write_rds(crash_with_clusters,
          "output/processed/crash_with_clusters.rds")
write_csv(crash_with_clusters,
          "output/processed/crash_with_clusters.csv")

message("✅ K-means analysis complete: outputs written to output/processed/ and plots to output/plots/")
