# %%
import spacy
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import sys
import pickle
import plotnine as p9
import warnings
import importlib

from plotnine import *
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from matplotlib.pyplot import figure
from collections import Counter
from matplotlib import pyplot as plt
from spacytextblob.spacytextblob import SpacyTextBlob

warnings.filterwarnings('ignore')
nltk.download('punkt')
pd.options.plotting.backend = "plotly"
sys.path.insert(0, 'C:/Users/Hubert/Dropbox/NLP/Project/eda_for_nlp_package.py')

#%%
print('Loading eda package...', flush = True)
import eda_for_nlp_package as eda
importlib.reload(eda)

print('Loading en pipeline...', flush = True)
en = spacy.load("en_core_web_md")
en.add_pipe('spacytextblob')
#%%
print('Loading data...', flush = True)
df = pd.read_csv('opensources_fake_news_cleaned.csv', nrows=100000)

# %%
print('Balancing classes...', flush = True)
unique_values = df['type'].unique()
filtered_df   = pd.DataFrame()

for value in unique_values:
    temp_df     = df[df['type'] == value].sample(10)
    filtered_df = pd.concat([filtered_df, temp_df])

df = filtered_df
df = df.reset_index(drop=True)

# %%
print('Removing numbers and \\n ...', flush = True)
df['content']=df['content'].apply(lambda x : eda.remove_num(x))
df['content']=df['content'].apply(lambda x : eda.remove_backslash_n(x))
#%%
print('Removing negation stopwords...', flush = True)
stopwords = en.Defaults.stop_words
stopwords = stopwords.difference({'except', 'however', "n't", 'no', 'nobody', 'none', 'noone', 'nor', 
                                           'not', 'nothing', 'n‘t', 'n’t',})
en.Defaults.stop_words = stopwords
# %%
print('Tokenization...', flush = True)
docs = eda.tokenize(df, en, 'content')

print('Saving tokens and en pipeline...', flush = True)
pickle.dump(en, open("en_dump.pickle", "wb"))
pickle.dump(docs, open("docs_dump.pickle", "wb"))

# %%
print('Adding stats to df...', flush = True)
df = eda.add_stats(df, docs)

# %%
print('Extracting nouns, noun chunks, enitities, and lemmas per document...', flush = True)
nouns_list        = eda.get_nouns_list(docs)
noun_chunk_list   = eda.get_noun_chunks_list(docs, stopwords=True)
entities_list     = eda.organisation_like_entitites_list(docs, uniq=True)
lemmas_list       = eda.get_lemmas_list(docs)
df['nouns']       = nouns_list
df['noun_chunks'] = noun_chunk_list
df['entities']    = entities_list
df['lemmas']      = lemmas_list
df
# %%
print('Calculating tf-idf scores, and preparation of top tf-idf table...', flush = True)
top_tfidf = eda.tfidf_table(df['content'], en, top=10)
top_tfidf.to_csv('top_tf-idf.csv')