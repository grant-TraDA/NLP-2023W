# Installing library for Quantitative Analysis of Textual Data (quanteda)
install.packages('quanteda')

# Loading libraries
library(quanteda)
library(quanteda.textstats)

# Loading the dataset
df <- read.csv('squad.txt', header = FALSE, col.names = c('text'))
# Changing data type to string
df$text <- as.character(df$text)

# Creating word corpus out of the dataset
crp <- corpus(df['text'])

# Tokenizing the corpus
tok <- tokens(crp, what = "word",
              remove_punct = TRUE,
              remove_symbols = TRUE,
              remove_numbers = TRUE,
              verbose = TRUE, 
              include_docvars = TRUE)
# Tokens to lowercase
tok <- tokens_tolower(tok)

# Computing readability metrics
readability <- textstat_readability(crp, c("meanSentenceLength","meanWordSyllables", "Flesch", "ARI"), remove_hyphens = TRUE,
                                    min_sentence_length = 1, max_sentence_length = 10000,
                                    intermediate = FALSE)
# Computing complexity metrics
complexity <- dfm(tok) %>% 
  textstat_lexdiv(measure = c("TTR", "CTTR", "D"))


# Saving metrics to a dedicated file
write.csv(readability, "readability.csv")
write.csv(complexity, "complexity.csv")







