from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class Wordcloud:

    def __init__(self):
        stopwords = set(STOPWORDS)
        stopwords.update(['a', 'i', 'u', 'e', 'o', 's', 't', 'm', 'd', 'n', 'r', 'l', 'c', 'p', 'g', 'h', 'b', 'f', 'k', 'w', 'v', 'y', 'j', 'z', 'x', 'q'])
        self.stopwords = stopwords

    def create_wordcloud(self, text, **kwargs):
        wordcloud = WordCloud(stopwords=self.stopwords, **kwargs).generate(text)

        return wordcloud
    
    def plot_wordcloud(self, text, title, **kwargs):
            
            wordcloud = self.create_wordcloud(text, **kwargs)
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.subplots_adjust(top=1.2) 
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(title, fontsize=32)
            ax.axis('off')
            plt.show()

class SentimentAnalyzer:

    def __init__(self):
          nltk.download('punkt')

    def get__textblob_sentiment(self, text):
        sentences = nltk.sent_tokenize(str(text))
        sentiment_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
        average_score = sum(sentiment_scores) / len(sentiment_scores)
        return average_score
     
    def get_vader_sentiment(self, text):
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)['compound']