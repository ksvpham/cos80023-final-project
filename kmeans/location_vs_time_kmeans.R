library(tidyr)
library(dplyr)
library(lubridate)
library(ggplot2)

# Read datasets
location <- read.csv("datasets/node.csv")
accident <- read.csv("datasets/accident.csv")

# Keep only the columns we need
location <- location[, c("ACCIDENT_NO", "DEG_URBAN_NAME")]

# Fill in values that have nothing - default to "none"
location <- location %>%
  mutate(
    DEG_URBAN_NAME = ifelse(DEG_URBAN_NAME == "" | is.na(DEG_URBAN_NAME),
                            "NONE",
                            DEG_URBAN_NAME)
  )

# Fill in the values
location$values <- 1

# Turning the rows into columns
location_wide <- location %>%
  pivot_wider(
    names_from = DEG_URBAN_NAME,
    values_from = values,
    values_fill = 0
  )
  
# Adding ACCIDENT_TIME from accident.csv
location_time <- location_wide %>%
  left_join(accident %>% select(ACCIDENT_NO, ACCIDENT_TIME),
            by = 'ACCIDENT_NO')

# Convert time to numeric hour
location_time <- location_time %>%
  mutate(
    ACCIDENT_TIME = lubridate::hms(ACCIDENT_TIME),
    ACCIDENT_TIME = hour(ACCIDENT_TIME) + minute(ACCIDENT_TIME) / 60
  )

# Collapse duplicates by time; one row per time with summed one-hots
surface_cols <- setdiff(names(location_time), c("ACCIDENT_NO", "ACCIDENT_TIME"))

location_grouped <- location_time %>%
  select(-ACCIDENT_NO) %>%
  group_by(ACCIDENT_TIME) %>%
  summarise(across(all_of(surface_cols), ~ sum(.x, na.rm = TRUE)), .groups = "drop") %>%
  arrange(ACCIDENT_TIME)

set.seed(42)

# Build numeric feature matrix
X <- location_grouped %>%
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
     main = "Elbow Plot (K-means on location + Time)")

# Run K-means with chosen k 
set.seed(42)

# Numeric feature matrix
X <- location_grouped %>%
  select(where(is.numeric)) %>%
  scale()

kmeansresult3 <- kmeans(X, centers = 4, nstart = 25)

# Add cluster labels back
location_grouped$cluster <- kmeansresult3$cluster

# Prepare data for plotting 
# Compute total number of accidents per time
location_grouped$total_accidents <- rowSums(location_grouped %>% select(-ACCIDENT_TIME, -cluster))

#  Plot 
ggplot(location_grouped, aes(x = ACCIDENT_TIME, y = total_accidents,
                             color = as.factor(cluster))) +
  geom_point(size = 2) +
  geom_line(aes(group = cluster), alpha = 0.4) +
  scale_x_continuous(breaks = seq(0, 24, 2)) +
  theme_minimal() +
  labs(
    title = "Accident Frequency by Time of Day (Location + Time Clusters)",
    x = "Time of Day (Hour)",
    y = "Number of Accidents",
    color = "Cluster"
  )

# Summary of each cluster
cluster_summary_loc <- location_grouped %>%
  group_by(cluster) %>%
  summarise(
    total_points = n(),
    avg_time = mean(ACCIDENT_TIME, na.rm = TRUE),
    avg_accidents = mean(total_accidents, na.rm = TRUE),
    max_accidents = max(total_accidents, na.rm = TRUE),
    min_accidents = min(total_accidents, na.rm = TRUE),
    .groups = "drop"
  )

cluster_summary_loc

# "location" columns (here: all one-hot location columns)
location_cols <- setdiff(names(location_grouped), c("ACCIDENT_TIME", "cluster", "total_accidents"))

location_summary <- location_grouped %>%
  group_by(cluster) %>%
  summarise(across(all_of(location_cols), ~ sum(.x, na.rm = TRUE)), .groups = "drop")

location_summary

# Combine it all together
summary_combined_loc <- cluster_summary_loc %>%
  left_join(location_summary, by = "cluster")

summary_combined_loc


