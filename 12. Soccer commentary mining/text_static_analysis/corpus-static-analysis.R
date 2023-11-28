library("quanteda")
library("quanteda.textstats")
library("ggplot2")

data <- read.csv("output.csv")

# Lets divide sentences recorded by whisper into matches. For the purpose of this
# analysis we treat each match collectively as separate text.
matches_text <- vapply(unique(data$file), function(x) {
  paste0(data[data$file == x,]$text, collapse = " ")
}, character(1))

matches <- data.frame(
  match = unique(data$file),
  text = matches_text
)

# Let's create corpus object

mycorpus <- corpus(matches, text_field = "text")

# We tokenize dataset to remove signs that could potentially fuzz the analysis

tok <- tokens(
  mycorpus,
  what = "word",
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_numbers = TRUE,
  remove_url = TRUE,
  remove_hyphens = FALSE,
  verbose = TRUE,
  include_docvars = TRUE
)

tok <- tokens_tolower(tok)

# Part of the static text analysis is cmaprsion of tokens with and without stopwords
# hence we create an alterantive set wit hstop words removed
tok_wo_stop <- tokens_select(tok, stopwords("english"), selection = "remove", padding = FALSE)
tokens <- data.frame(
  ntoken = c(ntoken(tok), ntoken(tok_wo_stop)),
  ntype = c(ntype(tok), ntype(tok_wo_stop)),
  stopwords = c(rep(FALSE, times = length(ntoken(tok))), rep(TRUE, times = length(ntoken(tok_wo_stop))))
)

ggplot(data = tokens) +
  geom_density(aes(x = ntoken, fill = stopwords), alpha = 0.5) +
  theme_bw()

ggplot(data = tokens) +
  geom_density(aes(x = ntype, fill = stopwords), alpha = 0.5) +
  theme_bw()

# We can observe line-ish realtionship between number of tokens and types 
# in the matches dataset

ggplot(data = tokens, aes(x = ntoken, y = ntype, color = stopwords)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y~x) +
  theme_bw()

ggplot(data = tokens, aes(x = ntoken, y = ntype, color = stopwords)) +
  geom_point() +
  geom_smooth() +
  theme_bw()

summary(lm(ntype~ntoken, data = tokens[tokens$stopwords, ]))

# Residuals:
#   Min      1Q  Median      3Q     Max 
# -548.96  -64.12   30.55  103.34  276.81 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 1.809e+02  1.226e+01   14.76   <2e-16 ***
#   ntoken      2.662e-01  4.596e-03   57.92   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 144.6 on 936 degrees of freedom
# Multiple R-squared:  0.7818,	Adjusted R-squared:  0.7816 
# F-statistic:  3354 on 1 and 936 DF,  p-value: < 2.2e-16

summary(lm(ntype~ntoken, data = tokens[!tokens$stopwords, ]))

# Residuals:
#   Min      1Q  Median      3Q     Max 
# -507.16  -80.46   35.04  116.44  335.48 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 2.562e+02  1.296e+01   19.77   <2e-16 ***
#   ntoken      1.410e-01  2.448e-03   57.61   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 156 on 936 degrees of freedom
# Multiple R-squared:   0.78,	Adjusted R-squared:  0.7798 
# F-statistic:  3318 on 1 and 936 DF,  p-value: < 2.2e-16


# Calculating lexical readability metrics

matches_stats <- textstat_readability(mycorpus, c("meanSentenceLength","meanWordSyllables", "Flesch.Kincaid", "Flesch"), remove_hyphens = TRUE,
                     min_sentence_length = 1, max_sentence_length = 100,
                     intermediate = FALSE)

scores <- lapply(c("meanSentenceLength","meanWordSyllables", "Flesch.Kincaid", "Flesch"), function(x) {
  x_values <- matches_stats[[x]]
  x_values <- x_values[!is.na(x_values)]
  data.frame(score = x_values, name = x)
})

df_scores <- do.call(rbind, scores[1:2])

ggplot(data = scores[[1]]) +
  geom_density(aes(x = score, fill = name), alpha = 0.5) +
  theme_bw()

ggplot(data = scores[[2]]) +
  geom_density(aes(x = score, fill = name), alpha = 0.5) +
  theme_bw()
# Flesch.Kincaid score - the lower score value is, the simplest text is
ggplot(data = scores[[3]]) +
  geom_density(aes(x = score, fill = name), alpha = 0.5) +
  theme_bw()
# Flesch score - the higher score value is, the simplest text is
ggplot(data = scores[[4]]) +
  geom_density(aes(x = score, fill = name), alpha = 0.5) +
  theme_bw()

# Calculating lexical richness metrics

# Calculting token to types ratio

ttr_sw <- textstat_lexdiv(dfm(tok), measure = "TTR")$TTR
ttr_wo_sw <- textstat_lexdiv(dfm(tok_wo_stop), measure = "TTR")$TTR
ttr_df <- data.frame(
  ttr = c(ttr_sw, ttr_wo_sw),
  stopwords = c(rep(TRUE, times = length(ttr_sw)), rep(FALSE, times = length(ttr_wo_sw)))
)

ggplot(data = ttr_df) +
  geom_density(aes(x = ttr, fill = stopwords), alpha = 0.5) +
  theme_bw()

# Calculating Hapax score - https://en.wikipedia.org/wiki/Hapax_legomenon
# We skip values above 0.8 as obvious outliers
calculte_hapax_score <- function(tok, cutoff = 0.8) {
  dfm_obj <- dfm(tok)
  res <- rowSums(dfm_obj == 1)/ntoken(dfm_obj)
  res[res < 0.8]
}

hapax_sw <- calculte_hapax_score(tok)
hapax_wo_sw <- calculte_hapax_score(tok_wo_stop)
hapax_df <- data.frame(
  hapax = c(hapax_sw, hapax_wo_sw),
  stopwords = c(rep(TRUE, times = length(hapax_sw)), rep(FALSE, times = length(hapax_wo_sw)))
)

ggplot(data = hapax_df) +
  geom_density(aes(x = hapax, fill = stopwords), alpha = 0.5) +
  theme_bw()


