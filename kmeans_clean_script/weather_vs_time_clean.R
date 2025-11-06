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
# Load and keep only weather + time-of-day fields
# ------------------------------------------------------------
data <- read_csv("output/processed/cluster_scaled.csv", show_col_types = FALSE)

# Identify columns to keep
time_cols    <- grep("^time_of_day", names(data), value = TRUE)
weather_cols <- grep("^weather", names(data), value = TRUE)

keep_cols <- c("accident_no", "accident_hour", time_cols, weather_cols)
df <- data %>% select(all_of(keep_cols))

# Safety check
stopifnot(length(time_cols) > 0, length(weather_cols) > 0)

# ------------------------------------------------------------
# Elbow plot (k = 1..10) on full data (weather + time only)
# ------------------------------------------------------------
set.seed(42)

X <- df %>%
  select(-accident_no) %>%        # drop ID
  select(where(is.numeric)) %>%   # keep numeric (all are already scaled)
  as.matrix()

k_max <- 10
wss <- sapply(1:k_max, function(k) {
  kmeans(X, centers = k, nstart = 5, iter.max = 100)$tot.withinss
})

plot(1:k_max, wss, type = "b", pch = 19,
     xlab = "Number of clusters (k)",
     ylab = "Total within-cluster sum of squares (WSS)",
     main = "Elbow Plot (K-means on Weather + Time-of-day)")

# ------------------------------------------------------------
# Fit K-means (set best_k after elbow; e.g., 2 or 3 or 4)
# ------------------------------------------------------------
best_k <- 4  
set.seed(42)
km <- kmeans(X, centers = best_k, nstart = 25, iter.max = 300)

df$cluster <- factor(km$cluster)

# ------------------------------------------------------------
# Derive a categorical Time-of-day label for plotting
# ------------------------------------------------------------
# If your data has only Morning/Afternoon/Evening, this will pick among them.
# If you have Night too, include its column in time_cols.
time_mat <- as.matrix(df[, time_cols, drop = FALSE])
td_idx   <- max.col(time_mat, ties.method = "first")
td_lab   <- colnames(time_mat)[td_idx]

df$time_of_day_label <- factor(td_lab,
                               levels = c("time_of_dayMorning",
                                          "time_of_dayAfternoon",
                                          "time_of_dayEvening"),
                               labels = c("Morning", "Afternoon", "Evening"))

# ------------------------------------------------------------
# Plot: frequency by time-of-day, coloured by cluster
# ------------------------------------------------------------
ggplot(df, aes(x = time_of_day_label, fill = cluster)) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  labs(title = "Accident Frequency by Time of Day (Weather + Time Clusters)",
       x = "Time of Day", y = "Number of Accidents", fill = "Cluster")

# ------------------------------------------------------------
# Cluster summary (like your original)
# ------------------------------------------------------------
cluster_summary <- df %>%
  group_by(cluster) %>%
  summarise(
    total_points = n(),
    avg_time = mean(accident_hour, na.rm = TRUE),
    morning_share   = mean(!!sym(time_cols[grepl("Morning", time_cols)]), na.rm = TRUE),
    afternoon_share = mean(!!sym(time_cols[grepl("Afternoon", time_cols)]), na.rm = TRUE),
    evening_share   = mean(!!sym(time_cols[grepl("Evening", time_cols)]), na.rm = TRUE),
    .groups = "drop"
  )

print(cluster_summary)

# ------------------------------------------------------------
# 🌦️ Weather composition per cluster (sum of scaled one-hots)
# ------------------------------------------------------------
weather_summary <- df %>%
  group_by(cluster) %>%
  summarise(across(all_of(weather_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
# Using mean z-scores so you can see above/below-average prevalence

print(weather_summary)

# ------------------------------------------------------------
# Combine (like your summary_combined)
# ------------------------------------------------------------
summary_combined <- cluster_summary %>%
  left_join(weather_summary, by = "cluster")

print(summary_combined)


