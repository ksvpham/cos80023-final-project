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
# Load and keep only WEATHER + time-of-day fields
# ------------------------------------------------------------
data <- read_csv("output/processed/cluster_scaled.csv", show_col_types = FALSE)

# Identify columns to keep
time_cols     <- grep("^time_of_day", names(data), value = TRUE)
weather_cols  <- grep("^weather", names(data), value = TRUE)

keep_cols <- c("accident_no", "accident_hour", time_cols, weather_cols)
df <- data %>% select(all_of(keep_cols))

# Safety check
stopifnot(length(time_cols) > 0, length(weather_cols) > 0)

# ------------------------------------------------------------
# ---- Elbow Plot (K-means on Weather + Time) ----
# ------------------------------------------------------------
set.seed(42)

# Build numeric feature matrix (drop zero-variance columns to prevent NaN errors)
X <- df %>%
  select(-accident_no) %>%
  select(where(is.numeric)) %>%
  select(where(~ sd(.x, na.rm = TRUE) > 0)) %>%
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
     main = "Elbow Plot (K-means on Weather + Time)")

# ------------------------------------------------------------
# ---- K-means on Weather + Time (set k) ----
# ------------------------------------------------------------
set.seed(42)
wg <- df %>%
  # work on rows with valid time only (accident_hour is already numeric/scaled)
  filter(!is.na(accident_hour))

# Build numeric feature matrix and drop zero-variance columns
X_w <- wg %>%
  select(-accident_no) %>%
  select(where(is.numeric)) %>%
  select(where(~ sd(.x, na.rm = TRUE) > 0)) %>%
  scale()

# Final guard: ensure no NA/NaN/Inf remain
if (any(!is.finite(X_w))) {
  stop("Non-finite values remain in feature matrix after filtering/scaling.")
}

best_k <- 4
kmeans_w <- kmeans(X_w, centers = best_k, nstart = 25)

# Attach clusters back
wg$cluster <- factor(kmeans_w$cluster)

# ------------------------------------------------------------
# Derive time-of-day label for bar plot (Morning / Afternoon / Evening / Night)
# ------------------------------------------------------------
time_mat <- as.matrix(wg[, time_cols, drop = FALSE])
td_idx   <- max.col(time_mat, ties.method = "first")
td_lab   <- colnames(time_mat)[td_idx]

levels_in_data <- gsub("^time_of_day", "", unique(td_lab))
ordered_levels <- c("Morning", "Afternoon", "Evening", "Night")
levels_present <- intersect(ordered_levels, levels_in_data)

wg$time_of_day_label <- factor(
  gsub("^time_of_day", "", td_lab),
  levels = levels_present,
  labels = levels_present
)

# ------------------------------------------------------------
# Bar Plot: Frequency by Time-of-Day (Weather + Time Clusters)
# ------------------------------------------------------------
ggplot(wg, aes(x = time_of_day_label, fill = cluster)) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  labs(
    title = "Accident Frequency by Time of Day (Weather + Time Clusters)",
    x = "Time of Day",
    y = "Number of Accidents",
    fill = "Cluster"
  )

# ------------------------------------------------------------
# Summaries (same structure as your weather version) + NIGHT
# ------------------------------------------------------------

# Helper to pull a mean share for a specific time label if present
mean_or_na <- function(.data, pat) {
  col <- time_cols[grepl(pat, time_cols)]
  if (length(col) == 0) return(NA_real_)
  mean(.data[[col]], na.rm = TRUE)
}

cluster_summary <- wg %>%
  group_by(cluster) %>%
  summarise(
    total_points    = n(),
    avg_time        = mean(accident_hour, na.rm = TRUE),
    morning_share   = mean_or_na(cur_data_all(), "Morning"),
    afternoon_share = mean_or_na(cur_data_all(), "Afternoon"),
    evening_share   = mean_or_na(cur_data_all(), "Evening"),
    night_share     = mean_or_na(cur_data_all(), "Night"),
    .groups = "drop"
  )

print(cluster_summary)

# Weather composition per cluster (mean z-scores)
weather_summary <- wg %>%
  group_by(cluster) %>%
  summarise(across(all_of(weather_cols), ~ mean(.x, na.rm = TRUE)),
            .groups = "drop")
# (Means of scaled one-hots show above/below-average prevalence)

print(weather_summary)

# ------------------------------------------------------------
# Combine (like your summary_combined)
# ------------------------------------------------------------
summary_combined <- cluster_summary %>%
  left_join(weather_summary, by = "cluster")

print(summary_combined)

