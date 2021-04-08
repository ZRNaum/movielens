#Q1
dim(edx)


#Q2
edx %>% filter(rating == 0) %>% tally()
edx %>% filter(rating == 3) %>% tally()


#Q3
n_distinct(edx$movieId)


#Q4
n_distinct((edx$userId))


#Q5
sapply(c("Drama", "Comedy", "Thriller", "Romance"), function(g) {
  sum(str_detect(edx$genres, g))
})


#Q6
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))


#Q7
edx %>% group_by(rating) %>%
  summarize(n = n()) %>%
  slice_max(n, n = 5) %>%
  arrange(desc(n))


#Q8
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()