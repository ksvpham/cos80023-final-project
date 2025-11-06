# ------------------------------------------------------------
# 📦 Libraries
# ------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(stringr)
library(forcats)

# ------------------------------------------------------------
# Load and keep only ROAD SURFACE + time-of-day fields
# ------------------------------------------------------------
data <- read_csv("output/processed/cluster_scaled.csv", show_col_types = FALSE)

# Identify columns to keep
time_cols        <- grep("^time_of_day", names(data), value = TRUE)
road_surface_cols <- grep("^road_surface", names(data), value = TRUE)

keep_cols <- c("accident_no", "accident_hour", time_cols, road_surface_cols)
df <- data %>% select(all_of(keep_cols))

# Safety check
stopifnot(length(time_cols) > 0, length(road_surface_cols) > 0)

# ------------------------------------------------------------
# Elbow plot (k = 1..10) on full data (road surface + time only)
# ------------------------------------------------------------
set.seed(42)

X <- df %>%
  select(-accident_no) %>%        # drop ID
  select(where(is.numeric)) %>%   # keep numeric (scaled)
  as.matrix()

k_max <- 10
wss <- sapply(1:k_max, function(k) {
  kmeans(X, centers = k, nstart = 5, iter.max = 100)$tot.withinss
})

plot(1:k_max, wss, type = "b", pch = 19,
     xlab = "Number of clusters (k)",
     ylab = "Total within-cluster sum of squares (WSS)",
     main = "Elbow Plot (K-means on Road Surface + Time-of-day)")

# ------------------------------------------------------------
# Fit K-means 
# ------------------------------------------------------------
best_k <- 3
set.seed(42)
km <- kmeans(X, centers = best_k, nstart = 25, iter.max = 300)

df$cluster <- factor(km$cluster)

# ------------------------------------------------------------
# Derive a categorical Time-of-day label for plotting
# ------------------------------------------------------------
# If your data has only Morning/Afternoon/Evening, this will pick among them.
# If you have Night too, include its column in time_cols and it will be used.
time_mat <- as.matrix(df[, time_cols, drop = FALSE])
td_idx   <- max.col(time_mat, ties.method = "first")
td_lab   <- colnames(time_mat)[td_idx]

# Create clean labels based on detected columns
levels_in_data <- gsub("^time_of_day", "", unique(td_lab))
ordered_levels <- c("Morning","Afternoon","Evening","Night")
levels_present <- intersect(ordered_levels, levels_in_data)

df$time_of_day_label <- factor(
  gsub("^time_of_day", "", td_lab),
  levels = levels_present,
  labels = levels_present
)

# ------------------------------------------------------------
# Plot: frequency by time-of-day, coloured by cluster
# ------------------------------------------------------------
ggplot(df, aes(x = time_of_day_label, fill = cluster)) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  labs(title = "Accident Frequency by Time of Day (Road Surface + Time Clusters)",
       x = "Time of Day", y = "Number of Accidents", fill = "Cluster")

# ------------------------------------------------------------
# Cluster summary (like your original)
# ------------------------------------------------------------
# Helper to pull a mean share for a specific time label if present
mean_or_na <- function(.data, pat) {
  col <- time_cols[grepl(pat, time_cols)]
  if (length(col) == 0) return(NA_real_)
  mean(.data[[col]], na.rm = TRUE)
}

cluster_summary <- df %>%
  group_by(cluster) %>%
  summarise(
    total_points = n(),
    avg_time     = mean(accident_hour, na.rm = TRUE),
    morning_share   = mean_or_na(cur_data_all(), "Morning"),
    afternoon_share = mean_or_na(cur_data_all(), "Afternoon"),
    evening_share   = mean_or_na(cur_data_all(), "Evening"),
    night_share     = mean_or_na(cur_data_all(), "Night"),
    .groups = "drop"
  )

print(cluster_summary)

# ------------------------------------------------------------
# Road surface composition per cluster (mean z-scores)
# ------------------------------------------------------------
road_surface_summary <- df %>%
  group_by(cluster) %>%
  summarise(across(all_of(road_surface_cols), ~ mean(.x, na.rm = TRUE)),
            .groups = "drop")
# (Means of scaled one-hots show above/below-average prevalence for each surface)

print(road_surface_summary)

# ------------------------------------------------------------
# Combine (like your summary_combined)
# ------------------------------------------------------------
summary_combined_rs <- cluster_summary %>%
  left_join(road_surface_summary, by = "cluster")

print(summary_combined_rs)
