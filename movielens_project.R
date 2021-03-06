##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(recommenderlab)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
### End of provided code ###


# Mutate edx and validation datasets to pull date from timestamp field
edx <- edx %>% mutate(timestamp = as_datetime(timestamp),
                      year = year(ymd_hms(timestamp)),
                      month = month(ymd_hms(timestamp)),
                      day = day(ymd_hms(timestamp)))
validation <- validation %>% mutate(timestamp = as_datetime(timestamp),
                                    year = year(ymd_hms(timestamp)),
                                    month = month(ymd_hms(timestamp)),
                                    day = day(ymd_hms(timestamp)))

# Export datasets to .csv for use in report
write_csv(edx, "edx.csv")
write_csv(validation, "validation.csv")

# Generate train and test sets from edx dataset
set.seed(1, sample.kind="Rounding")
index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-index]
test_set <- edx[index]
test_set <- test_set %>%
  semi_join(train_set, by = 'movieId') %>%
  semi_join(train_set, by = 'userId')

# Compute average ratings for various predictors to determine variability
# UserId
train_set %>% 
  group_by(userId) %>% 
  filter(n()>=100) %>%
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Genres
train_set %>%
  group_by(genres) %>%
  filter(n() >= 1000) %>%
  summarize(b_g = mean(rating)) %>%
  ggplot(aes(b_g)) +
  geom_histogram(bins = 30, color = "black")

# Year
train_set %>%
  group_by(year) %>%
  filter(n() >= 5) %>%
  summarize(b_y = mean(rating)) %>%
  ggplot(aes(b_y)) +
  geom_histogram(bins = 30, color = "black")

# Month
train_set %>%
  group_by(month) %>%
  summarize(b_m = mean(rating)) %>%
  ggplot(aes(b_m)) +
  geom_histogram(bins = 30, color = "black")

# Day
train_set %>%
  group_by(day) %>%
  summarize(b_d = mean(rating)) %>%
  ggplot(aes(b_d)) +
  geom_histogram(bins = 30, color = "black")

# Testing model to find the lambda that minimizes the RMSE
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l) {
  mu_hat <- mean(train_set$rating)
  
  # Movie bias
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n() + l))
  
  # User bias
  b_u <- train_set %>%
    left_join(b_i, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_hat - b_i)/(n() + l))
  
  # Genre bias
  b_g <- train_set %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu_hat - b_i - b_u)/(n() + l))
  
  # Day bias
  b_d <- train_set %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    left_join(b_g, by = 'genres') %>%
    group_by(day) %>%
    summarize(b_d = sum(rating - mu_hat - b_i - b_u - b_g)/(n() + l))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    left_join(b_g, by = 'genres') %>%
    left_join(b_d, by = 'day') %>%
    mutate(pred = mu_hat + b_i + b_u + b_g + b_d) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})

# Best lambda value
lambda <- lambdas[which.min(rmses)]


# Final RMSE calculation using validation set
final_rmse <- sapply(lambda, function(l) {
  mu_hat <- mean(edx$rating)
  
  # Movie bias
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n() + l))
  
  # User bias
  b_u <- edx %>%
    left_join(b_i, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_hat - b_i)/(n() + l))
  
  # Genre bias
  b_g <- edx %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu_hat - b_i - b_u)/(n() + l))
  
  # Day bias
  b_d <- edx %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    left_join(b_g, by = 'genres') %>%
    group_by(day) %>%
    summarize(b_d = sum(rating - mu_hat - b_i - b_u - b_g)/(n() + l))
  
  predicted_ratings <- validation %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    left_join(b_g, by = 'genres') %>%
    left_join(b_d, by = 'day') %>%
    mutate(pred = mu_hat + b_i + b_u + b_g + b_d) %>%
    pull(pred)
  
  return(RMSE(validation$rating, predicted_ratings))
})