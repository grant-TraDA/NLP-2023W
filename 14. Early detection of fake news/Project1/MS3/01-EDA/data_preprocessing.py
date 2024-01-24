import spacy
import pandas as pd
import os
import typer
import sys
import pickle
import warnings
import importlib

from pathlib import Path
from plotnine import *
from spacytextblob.spacytextblob import SpacyTextBlob


#sys.path.insert(0, 'C:/Users/Hubert/Dropbox/NLP/Project/eda_for_nlp_package.py')
# data_path = 'opensources_fake_news_cleaned.csv'
def main(eda_for_nlp_path : str, output_dir: str, data_path: str, obs_per_class: int = 10, top_tf_idf: int = 10):

    print('Loading eda package and options setup...', flush = True)
    sys.path.insert(0, eda_for_nlp_path)
    import eda_for_nlp_package as eda
    importlib.reload(eda)
    warnings.filterwarnings('ignore')
    pd.options.plotting.backend = "plotly"


    print('Loading en pipeline...', flush = True)
    en = spacy.load("en_core_web_md")
    en.add_pipe('spacytextblob')


    print('Removing negation stopwords...', flush = True)
    stopwords = en.Defaults.stop_words
    stopwords = stopwords.difference({'except', 'however', "n't", 'no', 'nobody', 
                                      'none', 'noone', 'nor', 'not', 'nothing', 
                                      'n‘t', 'n’t',})
    en.Defaults.stop_words = stopwords


    print('Loading data...', flush = True)
    df = pd.read_csv(data_path, nrows = obs_per_class * 1000)


    print('Changing to output dir...', flush = True)
    try:
        os.chdir(output_dir)
    except:
        Path(output_dir).mkdir(parents = True, exist_ok = True)
        os.chdir(output_dir)
    print(os.getcwd(), flush = True)


    print('Balancing classes...', flush = True)
    unique_values = df['type'].unique()
    filtered_df   = pd.DataFrame()
    for value in unique_values:
        temp_df     = df[df['type'] == value].sample(obs_per_class)
        filtered_df = pd.concat([filtered_df, temp_df])
    df = filtered_df
    df = df.reset_index(drop = True)


    print('Removing numbers and \\n ...', flush = True)
    df['content']=df['content'].apply(lambda x : eda.remove_num(x))
    df['content']=df['content'].apply(lambda x : eda.remove_backslash_n(x))


    print('Tokenization...', flush = True)
    docs = eda.tokenize(df, en, 'content')


    print('Saving tokens and en pipeline...', flush = True)
    pickle.dump(en, open("en.pickle", "wb"))
    pickle.dump(docs, open("docs.pickle", "wb"))


    print('Adding stats to df...', flush = True)
    df = eda.add_stats(df, docs)


    print('Extracting nouns, noun chunks, enitities, and lemmas per document...', flush = True)
    nouns_list        = eda.get_nouns_list(docs)
    noun_chunk_list   = eda.get_noun_chunks_list(docs, stopwords=True)
    entities_list     = eda.organisation_like_entitites_list(docs, uniq=True)
    lemmas_list       = eda.get_lemmas_list(docs)
    df['nouns']       = nouns_list
    df['noun_chunks'] = noun_chunk_list
    df['entities']    = entities_list
    df['lemmas']      = lemmas_list


    print('Saving the main df...', flush = True)
    df.to_csv('main_df.csv')


    print('Calculating tf-idf scores, and preparation of top tf-idf table...', flush = True)
    top_tfidf = eda.tfidf_table(df['content'], en, top = top_tf_idf)
    top_tfidf.to_csv('top_tf-idf.csv')

if __name__=="__main__":
    typer.run(main)