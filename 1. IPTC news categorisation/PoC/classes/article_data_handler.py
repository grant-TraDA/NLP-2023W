import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from classes.embeddings import BertEmbedding, Word2VecEmbedding
from scipy import stats
from scipy.stats import kruskal
from sklearn.cluster import KMeans, DBSCAN

class ArticleDataHandler:
    """Class for handling article data."""
    
    def __init__(self, dataframe, text_column='text', category_column='categories', embedding_column='bert_embedding'):
        
        """Initializes the ArticleDataHandler class.
        Args:
            dataframe: The dataframe containing the article data.
            embedding: The embedding to use for the articles (bert or word2vec).
            text_column: The column containing the text.
            category_column: The column containing the categories.
        """

        self.data = dataframe
        self.text_column = text_column
        self.category_column = category_column
        self.embedding_column = embedding_column
    
        self.embeddings = dataframe[self.embedding_column].apply(eval)
        self.embeddings = pd.DataFrame(self.embeddings.tolist())

    def get_outliers_indices(self, method='lof', threshold=3):
        
        """Returns the outliers in the data.
        Args:
            method: The method used for outlier detection (lof, svm or iforest)
        Returns:
            The outliers in the data.
        """

        if method == 'lof':
            return np.where(LocalOutlierFactor().fit_predict(self.embeddings) == -1)
        elif method == 'svm':
            return np.where(OneClassSVM().fit_predict(self.embeddings) == -1)
        elif method == 'iforest':
            return np.where(IsolationForest().fit_predict(self.embeddings) == -1)
        elif method == 'zscore':
            z = np.abs(stats.zscore(self.embeddings))
            return np.where(z > threshold)
        else:
            raise ValueError('method must be either lof, svm, iforest or zscore')

    def get_cluster_indices(self, method='kmeans', n_clusters=None):
            
            """Returns the clusters of the data.
            Args:
                method: The method used for clustering (kmeans or dbscan)
            Returns:
                The clusters in the data.
            """
            
            if not n_clusters:
                n_clusters = self.data[self.category_column].nunique()

            if method == 'kmeans':
                return KMeans(n_clusters=n_clusters).fit_predict(self.embeddings)
            elif method == 'dbscan':
                return DBSCAN().fit_predict(self.embeddings)
            else:
                raise ValueError('method must be either kmeans or dbscan')

    def article_count_by_category(self, category_column=None):
        
        """Returns the number of articles in each category.
        Args:
            category_column: The column containing the categories.
        Returns:
            The number of articles in each category.
        """
        if not category_column:
            category_column = self.category_column
        return self.data[category_column].value_counts()


    def tfidf_importance(self, text_column=None, category_column=None):
        """Returns the TF-IDF importance of each word in each category (in descending order).
        Args:
            text_column: The column containing the text.
            category_column: The column containing the categories.
        Returns:
            The TF-IDF importance of each word in each category.
        """
        if not text_column:
            text_column = self.text_column
        if not category_column:
            category_column = self.category_column

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data[text_column])
        categories = self.data[category_column]
        importance = {}

        feature_names = tfidf_vectorizer.get_feature_names_out()

        for category in categories.unique():
            category_indices = categories[categories == category].index
            category_tfidf = tfidf_matrix[category_indices]
            avg_tfidf = category_tfidf.mean(axis=0)
            avg_tfidf_list = avg_tfidf.tolist()[0]

            words_importance = {feature_names[i]: score for i, score in enumerate(avg_tfidf_list) if score > 0}
            importance[category] = {k: v for k, v in sorted(words_importance.items(), key=lambda item: item[1], reverse=True)}
            
        return importance
    

    def article_length_by_category(self, text_column=None, category_column=None):
        
        """Returns the average length of articles in each category.
        Args:
            text_column: The column containing the text.
            category_column: The column containing the categories.
        Returns:
            The average length of articles in each category.    
        """
        if not text_column:
            text_column = self.text_column
        if not category_column:
            category_column = self.category_column

        self.data['article_length'] = self.data[text_column].apply(lambda x: len(re.findall(r'\w+', x)))
        return self.data.groupby(category_column)['article_length'].mean()


    def get_sentiment_analysis(self, text_column=None):
        
        """
        Returns the sentiment of each article.
        Args:
            text_column: The column containing the text.
            Returns:
            The sentiment of each article.
        """
        if not text_column:
            text_column = self.text_column

        nltk.download('vader_lexicon')
        sentiment_analyzer = SentimentIntensityAnalyzer()
        self.data['sentiment'] = self.data[text_column].apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])
        return self.data['sentiment']

    def get_sentiment_by_category(self, text_column=None, category_column=None):
            
            """
            Returns the sentiment of each category.
            Args:
                text_column: The column containing the text.
                category_column: The column containing the categories.
            Returns:
                The sentiment of each category.
            """
            if not text_column:
                text_column = self.text_column
            if not category_column:
                category_column = self.category_column
    
            if 'sentiment' not in self.data.columns:
                self.get_sentiment_analysis(text_column=text_column)
            return self.data.groupby(category_column)['sentiment'].mean()

    def sentiment_difference_significance(self, category_column=None):
        
        """
        Uses Kruskal-Wallis H-test to test if the sentiment differs between the categories.
        Args:
            category_column: The column containing the categories.
        Returns:
            The Kruskal-Wallis H-test statistic and p-value.
        """
        if not category_column:
            category_column = self.category_column
            
        if 'sentiment' not in self.data.columns:
            self.get_sentiment_analysis(text_column='text')
        grouped_data = [self.data['sentiment'][self.data[category_column] == category] for category in self.data[category_column].unique()]

    
        statistic, pvalue = kruskal(*grouped_data)  
        return statistic, pvalue

    def get_distribution_of_categories_over_time(self, category_column=None, date_column=None):
        
        """
        Returns the distribution of categories over time.
        Args:
            category_column: The column containing the categories.
            date_column: The column containing the dates.
        Returns:
            The distribution of categories over time.
        """
        if not category_column:
            category_column = self.category_column
        if not date_column:
            date_column = 'date'

        return self.data.groupby([date_column, category_column]).size().unstack(fill_value=0)
