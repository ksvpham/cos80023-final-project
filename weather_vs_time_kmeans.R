library(tidyr)
library(dplyr)
library(lubridate)
library(ggplot2)

# Read dataset
atmospheric <- read.csv("datasets/atmospheric_cond.csv")
accident <- read.csv("datasets/accident.csv")

# Add a dummy value (e.g., 1) so we can fill the new columns
atmospheric$value <- 1

# Pivot the rows into columns
atmospheric_wide <- atmospheric %>%
  pivot_wider(
    names_from = ATMOSPH_COND_DESC,  # each condition becomes a column
    values_from = value,             # values to fill in cells
    values_fill = 0                  # fill missing with 0 instead of NA
  )

# Merge ACCIDENT_TIME into the wide dataset
atmospheric_wide <- atmospheric_wide %>%
  left_join(accident %>% select(ACCIDENT_NO, ACCIDENT_TIME),
            by = "ACCIDENT_NO")

# Convert ACCIDENT_TIME to numeric hour (e.g. 13:30 -> 13.5)
atmospheric_wide <- atmospheric_wide %>%
  mutate(
    ACCIDENT_TIME = lubridate::hms(ACCIDENT_TIME),
    ACCIDENT_TIME = hour(ACCIDENT_TIME) + minute(ACCIDENT_TIME) / 60
  )

# Remove unnecessary columns
atmospheric_wide <- atmospheric_wide %>%
  select(-c(ATMOSPH_COND, ATMOSPH_COND_SEQ))

# Collapse duplicates by time & weather; one row per time
# Identify the weather condition (one-hot) columns
cond_cols <- setdiff(names(atmospheric_wide), c("ACCIDENT_NO", "ACCIDENT_TIME"))

# Drop ACCIDENT_NO and sum the one-hot columns within each time
atmospheric_grouped <- atmospheric_wide %>%
  select(-ACCIDENT_NO) %>%
  group_by(ACCIDENT_TIME) %>%
  summarise(across(all_of(cond_cols), ~ sum(.x, na.rm = TRUE)), .groups = "drop") %>%
  arrange(ACCIDENT_TIME)

head(atmospheric_grouped)



set.seed(42)

# Build numeric feature matrix
X <- atmospheric_grouped %>%
  select(where(is.numeric)) %>%
  scale()

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


# Run K-means with chosen k 
set.seed(42)

# Numeric feature matrix
X <- atmospheric_grouped %>%
  select(where(is.numeric)) %>%
  scale()

kmeansresult1 <- kmeans(X, centers = 2, nstart = 25)

# Add cluster labels back
atmospheric_grouped$cluster <- kmeansresult1$cluster

# Prepare data for plotting 
# Compute total number of accidents per time
atmospheric_grouped$total_accidents <- rowSums(atmospheric_grouped %>% select(-ACCIDENT_TIME, -cluster))

#  Plot 
ggplot(atmospheric_grouped, aes(x = ACCIDENT_TIME, y = total_accidents,
                                color = as.factor(cluster))) +
  geom_point(size = 2) +
  geom_line(aes(group = cluster), alpha = 0.4) +
  scale_x_continuous(breaks = seq(0, 24, 2)) +
  theme_minimal() +
  labs(
    title = "Accident Frequency by Time of Day (Weather + Time Clusters)",
    x = "Time of Day (Hour)",
    y = "Number of Accidents",
    color = "Cluster"
  )

# Summary of each cluster

cluster_summary <- atmospheric_grouped %>%
  group_by(cluster) %>%
  summarise(
    total_points = n(),
    avg_time = mean(ACCIDENT_TIME, na.rm = TRUE),
    avg_accidents = mean(total_accidents, na.rm = TRUE),
    max_accidents = max(total_accidents, na.rm = TRUE),
    min_accidents = min(total_accidents, na.rm = TRUE),
    .groups = "drop"
  )

cluster_summary

# Weather conditions in each cluster

weather_cols <- setdiff(names(atmospheric_grouped), c("ACCIDENT_TIME", "cluster", "total_accidents"))

weather_summary <- atmospheric_grouped %>%
  group_by(cluster) %>%
  summarise(across(all_of(weather_cols), ~ sum(.x, na.rm = TRUE)), .groups = "drop")

weather_summary

# Combine it all together
summary_combined <- cluster_summary %>%
  left_join(weather_summary, by = "cluster")

summary_combined
