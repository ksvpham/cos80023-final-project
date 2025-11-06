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
# Load and keep only LOCATION (deg_urban*) + time-of-day fields
# ------------------------------------------------------------
data <- read_csv("output/processed/cluster_scaled.csv", show_col_types = FALSE)

# Identify columns to keep
time_cols      <- grep("^time_of_day", names(data), value = TRUE)
location_cols  <- grep("^deg_urban", names(data), value = TRUE)

keep_cols <- c("accident_no", "accident_hour", time_cols, location_cols)
df <- data %>% select(all_of(keep_cols))

# Safety check
stopifnot(length(time_cols) > 0, length(location_cols) > 0)

# ------------------------------------------------------------
# Elbow plot (k = 1..10) on full data (location + time only)
# ------------------------------------------------------------
set.seed(42)

X <- df %>%
  select(-accident_no) %>%        # drop ID
  select(where(is.numeric)) %>%   # keep numeric (scaled)
  as.matrix()
X <- X[is.finite(rowSums(X)), , drop = FALSE]

k_max <- 10
wss <- sapply(1:k_max, function(k) {
  kmeans(X, centers = k, nstart = 5, iter.max = 100)$tot.withinss
})

plot(1:k_max, wss, type = "b", pch = 19,
     xlab = "Number of clusters (k)",
     ylab = "Total within-cluster sum of squares (WSS)",
     main = "Elbow Plot (K-means on Location (deg_urban*) + Time-of-day)")

# ------------------------------------------------------------
# Fit K-means 
# ------------------------------------------------------------

best_k <- 3
set.seed(42)

keep_rows <- is.finite(rowSums(X))          # rows used in X
X_fit <- X[keep_rows, , drop = FALSE]       # X actually passed to kmeans

km <- kmeans(X_fit, centers = best_k, nstart = 25, iter.max = 300)

# assign clusters back to df, leaving NA where a row was dropped from X
df$cluster <- NA_integer_
df$cluster[keep_rows] <- km$cluster
df$cluster <- factor(df$cluster)
# ------------------------------------------------------------
# Derive a categorical Time-of-day label for plotting
# ------------------------------------------------------------
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
  labs(title = "Accident Frequency by Time of Day (Location + Time Clusters)",
       x = "Time of Day", y = "Number of Accidents", fill = "Cluster")

# ------------------------------------------------------------
# Cluster summary (time-of-day shares etc.)
# ------------------------------------------------------------
mean_or_na <- function(.data, pat) {
  col <- time_cols[grepl(pat, time_cols)]
  if (length(col) == 0) return(NA_real_)
  mean(.data[[col]], na.rm = TRUE)
}

cluster_summary_loc <- df %>%
  group_by(cluster) %>%
  summarise(
    total_points     = n(),
    avg_time         = mean(accident_hour, na.rm = TRUE),
    morning_share    = mean_or_na(cur_data_all(), "Morning"),
    afternoon_share  = mean_or_na(cur_data_all(), "Afternoon"),
    evening_share    = mean_or_na(cur_data_all(), "Evening"),
    night_share      = mean_or_na(cur_data_all(), "Night"),
    .groups = "drop"
  )

print(cluster_summary_loc)

# ------------------------------------------------------------
# Location composition per cluster (mean z-scores of deg_urban*)
# ------------------------------------------------------------
location_summary <- df %>%
  group_by(cluster) %>%
  summarise(across(all_of(location_cols), ~ mean(.x, na.rm = TRUE)),
            .groups = "drop")
# (Means of scaled one-hots show above/below-average prevalence for each location)

print(location_summary)

# ------------------------------------------------------------
# Combine (like your summary_combined)
# ------------------------------------------------------------
summary_combined_loc <- cluster_summary_loc %>%
  left_join(location_summary, by = "cluster")

print(summary_combined_loc)
