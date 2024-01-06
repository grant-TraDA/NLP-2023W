library(dplyr)
library(ggplot2)

# Output processing


# Some matches were identified to have no comment. We exclude them from the analysis
matches_wo_comment <- c(430L, 460L, 490L, 516L, 524L, 528L, 610L, 636L, 866L, 892L, 
  910L, 97L, 215L, 237L, 273L, 431L, 461L, 495L, 529L, 611L, 665L, 
  691L, 731L, 867L, 875L, 879L, 881L, 893L, 905L, 917L, 931L, 933L
)

# We prepared the results in 2 different formats, one with row per event and one with
# row per match, we will use both of them depending on the type of the plot.
data <- read.csv("sentiments_labels_merged_v2.csv")
data2 <- read.csv("../sentiments_labels_merged.csv")

data <- data[!data$match_id %in% matches_wo_comment, ]
data2 <- data2[!data2$match_id %in% matches_wo_comment, ]

data <- data[, !endsWith(colnames(data), "event_time")]
for (c in colnames(data)) {
  data[[c]][is.infinite(data[[c]])] <- -1
  data[[c]][is.na(data[[c]])] <- -1
}

# Converting numerical  vader scores to classes 

data$vader_tag <- c("Positive", "Negative", "Neutral")[apply(data[, c("vader_positive", "vader_negative", "vader_neutral")] , 1, which.max)]
data2$vader_tag <- c("Positive", "Negative", "Neutral")[apply(data2[, c("vader_positive", "vader_negative", "vader_neutral")] , 1, which.max)]


# Generating roll mean to account for the context of sentences.

data <- group_by(data, "match_id") |> mutate(
  flair_sentiment_ma_3 =zoo::rollmean(flair_sentiment, 3, fill = 0),
  flair_sentiment_ma_5 =zoo::rollmean(flair_sentiment, 5, fill = 0),
  flair_sentiment_ma_7 =zoo::rollmean(flair_sentiment, 7, fill = 0),
  vader_sentiment_ma_3 =zoo::rollmean(vader_compound, 3, fill = 0),
  vader_sentiment_ma_5 =zoo::rollmean(vader_compound, 5, fill = 0),
  vader_sentiment_ma_7 =zoo::rollmean(vader_compound, 7, fill = 0)
)

# Generating roll mean to account for the context of sentences.

data2 <- group_by(data2, "match_id") |> mutate(
  flair_sentiment_ma_3 =zoo::rollmean(flair_sentiment, 3, fill = 0),
  flair_sentiment_ma_5 =zoo::rollmean(flair_sentiment, 5, fill = 0),
  flair_sentiment_ma_7 =zoo::rollmean(flair_sentiment, 7, fill = 0),
  vader_sentiment_ma_3 =zoo::rollmean(vader_compound, 3, fill = 0),
  vader_sentiment_ma_5 =zoo::rollmean(vader_compound, 5, fill = 0),
  vader_sentiment_ma_7 =zoo::rollmean(vader_compound, 7, fill = 0)
)

### Flair and Vader bar plots for labels

# Data preparation 

data_sentiment_categorical <- data.frame(
  rbind(
    data.frame(
      sentiment = ifelse(data$flair_tag == "NEGATIVE", "Negative", "Positive"),
      label = "Flair"
    ),
    data.frame(
      sentiment = data$vader_tag,
      label = "Vader"
    )
  )
)

res <- as.data.frame(table(data_sentiment_categorical))

ggplot(res, mapping = aes(y = Freq, x = sentiment, fill = label)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_text(aes(label=Freq), vjust=-0.5, color="black",
            position = position_dodge(0.9), size=3.5) +
  scale_y_continuous(labels = scales::label_number_si()) +
  labs(x = "Sentiment", y = "Sentences [n]", title = "Sentiment extracted by Vader and Flair", fill = "Model") +
  theme_bw()


data_sentiment_categorical_adj <- data.frame(
  rbind(
    data.frame(
      sentiment = ifelse(data$flair_sentiment < -0.5, "Negative", if_else(data$flair_sentiment > 0.5, "Positive", "Neutral")), # Generating adjusted Falair score
      label = "Flair"
    ),
    data.frame(
      sentiment = data$vader_tag,
      label = "Vader"
    )
  )
)

res_adj <- as.data.frame(table(data_sentiment_categorical_adj))

ggplot(res_adj, mapping = aes(y = Freq, x = sentiment, fill = label)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_text(aes(label=Freq), vjust=-0.5, color="black",
            position = position_dodge(0.9), size=3.5) +
  scale_y_continuous(labels = scales::label_number_si()) +
  labs(x = "Sentiment", y = "Sentences [n]", title = "Sentiment extracted by Vader and Flair with adjusted labels", fill = "Model") +
  theme_bw()

### Flair and Vader density plot for scores

data_density <- rbind(
  data.frame(
    sentiment = data$flair_sentiment,
    label = "Flair"
  ),
  data.frame(
    sentiment = data$vader_compound,
    label = "Vader"
  )
)

ggplot(data_density, mapping = aes(x = sentiment, fill = label)) +
  geom_density(alpha = 0.5) +
  labs(x = "Sentiment score", y = "Density", fill = "Model", title = "Sentiment score density for Vader and Flair") +
  theme_bw()


ggplot(data_density, mapping = aes(x = sentiment, fill = label)) +
  geom_density(alpha = 0.5) +
  labs(x = "Sentiment score", y = "Density", fill = "Model", title = "Sentiment score density for Vader and Flair") +
  geom_vline(xintercept = c(-0.1, 0.1), size = 1) +
  theme_bw()

# 0-neighbourhoo desnity visualization

tmp <- data2[data2$vader_compound < -0.1 | data2$vader_compound > 0.1, ]
cut <- as.data.frame(round(table(tmp$label)/NROW(tmp)*100, 2))
whole <- as.data.frame(round(table(data2$label)/NROW(data2)*100, 2))
whole$label <- "Entire dataset"
cut$label <- "0-neighbourhood"

data_3 <- rbind(whole, cut)
data_3

ggplot(data_3, mapping = aes(y = Freq, x = Var1, fill = label)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_text(aes(label=Freq), hjust = -0.3,  color="black",
            position = position_dodge(0.9), size=3.5, angle = 90, parse = TRUE) +
  labs(x = "Event", y = "Ooccurance [%]", title = "Events occurance fraction accross subsets for Vader", fill = "Subset") +
  ylim(0, 80) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.85, hjust=1))
  

# Events distribution baseed on the snetimatent category

data4 <- as.data.frame(round(table(data2$label, ifelse(data2$flair_sentiment < -0.5, "Negative", if_else(data2$flair_sentiment > 0.5, "Positive", "Neutral")))/NROW(data2)*100, 2)) # Fraction of sentences with given adjusted Flair score

ggplot(data4, mapping = aes(y = Freq, x = Var1, fill = Var2)) +
  geom_bar(stat="identity") +
  labs(x = "Event", y = "Sentences fraction [%]", title = "Flair sentiment distribution for different events", fill = "Sentiment") +
  ylim(0, 70) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.85, hjust=1))

data5 <- as.data.frame(round(table(data2$label, data2$vader_tag)/NROW(data2)*100, 2))

ggplot(data5, mapping = aes(y = Freq, x = Var1, fill = Var2)) +
  geom_bar(stat="identity") +
  labs(x = "Event", y = "Sentences fraction [%]", title = "Vader sentiment distribution for different events", fill = "Sentiment") +
  ylim(0, 70) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.85, hjust=1))


# Moving average density plots

data8 <- rbind(
  data.frame(
    sentiment = data$flair_sentiment_ma_3,
    label = "3"
  ),
  data.frame(
    sentiment = data$flair_sentiment_ma_5,
    label = "5"
  ),
  data.frame(
    sentiment = data$flair_sentiment_ma_7,
    label = "7"
  )
)

ggplot(data8, mapping = aes(x = sentiment, fill = label)) +
  geom_density(alpha = 0.2) +
  labs(x = "Sentiment", y = "Density", title = "Flair sentiment score moving average density", fill = "k") +
  theme_bw()


ggplot(data8, mapping = aes(x = sentiment, fill = label)) +
  geom_density(alpha = 0.2) +
  labs(x = "Sentiment", y = "Density", title = "Flair sentiment score moving average density", fill = "k") +
  geom_vline(xintercept = c(-1, -3/5, -1/5, 1/5, 3/5, 1), size = 1) +
  theme_bw()


ggplot(data8[data8$label == "5", ], mapping = aes(x = sentiment)) +
  geom_density(fill = "#00BA38", alpha = 0.2) +
  labs(x = "Sentiment", y = "Density", title = "Flair sentiment score moving average (k=3) density") +
  geom_vline(xintercept = c(-0.8, 0.8), size = 1) +
  theme_bw()


### Peaks zoom-in plots

# Events comparison between entire datset and top peak

tmp2 <- data2[data2$flair_sentiment_ma_5 > 0.8, ]
cut <- as.data.frame(round(table(tmp2$label)/NROW(tmp2)*100, 2))
whole <- as.data.frame(round(table(data2$label)/NROW(data2)*100, 2))
whole$label <- "Entire dataset"
cut$label <- "Top peak"

data_13 <- rbind(whole, cut)
data_13

ggplot(data_13, mapping = aes(y = Freq, x = Var1, fill = label)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_text(aes(label=Freq), hjust = -0.3,  color="black",
            position = position_dodge(0.9), size=3.5, angle = 90, parse = TRUE) +
  labs(x = "Event", y = "Occurance [%]", title = "Events occurance fraction accross subsets for flair", fill = "Subset") +
  ylim(0, 80) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.85, hjust=1))

# Percentage change in events occurrence for dataset and top peak

tmp5 <- data2[data2$flair_sentiment_ma_5 > 0.8, ]
data_19 <- as.data.frame((round((table(tmp5$label)/NROW(tmp5))/(table(data2$label)/NROW(data2)), 3)-1)*100) # Calculating the percentage change between two sets of data


ggplot(data_19, mapping = aes(y = Freq, x = Var1)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_text(data = data_19[data_19$Freq > 0, ], aes(label=Freq), vjust=-0.6,  color="black", size=3.5, parse = TRUE) +
  geom_text(data = data_19[data_19$Freq < 0, ], aes(label=Freq), vjust=1.2,  color="black", size=3.5, parse = TRUE) +
  labs(x = "Event", y = "Difference [%]", title = "Difference in events distribution between top peak and entire dataset for flair", fill = "Model") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.85, hjust=1))

# Events comparison between entire dataset and bottom peak

tmp3 <- data2[data2$flair_sentiment_ma_5 < -0.8, ]
cut <- as.data.frame(round(table(tmp3$label)/NROW(tmp3)*100, 2))
whole <- as.data.frame(round(table(data2$label)/NROW(data2)*100, 2))
whole$label <- "Entire dataset"
cut$label <- "Bottom peak"

data_14 <- rbind(whole, cut)
data_14

ggplot(data_14, mapping = aes(y = Freq, x = Var1, fill = label)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_text(aes(label=Freq), hjust = -0.3,  color="black",
            position = position_dodge(0.9), size=3.5, angle = 90, parse = TRUE) +
  labs(x = "Event", y = "Occurance [%]", title = "Events occurance fraction accross subsets for flair", fill = "Subset") +
  ylim(0, 80) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.85, hjust=1))


# Percentage change in events occurrence for dataset and top peak

tmp4 <- data2[data2$flair_sentiment_ma_5 < -0.8, ]
data_18 <- as.data.frame((round((table(tmp4$label)/NROW(tmp4))/(table(data2$label)/NROW(data2)), 3)-1)*100) # Calculating the percentage change between two sets of data


ggplot(data_18, mapping = aes(y = Freq, x = Var1)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_text(data = data_18[data_18$Freq > 0, ], aes(label=Freq), vjust=-0.6,  color="black", size=3.5, parse = TRUE) +
  geom_text(data = data_18[data_18$Freq < 0, ], aes(label=Freq), vjust=1.2,  color="black", size=3.5, parse = TRUE) +
  labs(x = "Event", y = "Difference [%]", title = "Difference in events distribution between bot peak and entire dataset for flair", fill = "Model") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.85, hjust=1))
