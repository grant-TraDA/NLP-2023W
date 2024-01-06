data <- read.csv("sentiments_labels_merged.csv")

plot(density(data$flair_score))
table(data$flair_tag)
