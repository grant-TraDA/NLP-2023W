import spacy
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import re
import string
import textacy
from collections import Counter
from matplotlib import pyplot
import seaborn as sns
pd.options.plotting.backend = "plotly"
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from matplotlib.pyplot import figure
import nltk

en = spacy.load("en_core_web_md")

"""# Data preparation"""

def add_stats(df, docs):
    df['word_count']   = df["content"].apply(lambda x : len(x.split()))
    df['char_count']   = df['content'].apply(lambda x : len(x.replace(" ","")))
    df['word_density'] = df['word_count'] / (df['char_count'] + 1)
    sentiment          = count_sentiment(docs)
    df['polarity']     = sentiment['polarity']
    df['subjectivity'] = sentiment['subjectivity'] 
    return df

def my_lower(text):
    return text.lower()

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def remove_num(text):
    num  = re.compile(r'[0-9]{1,30}')
    text = num.sub(r'',text)
    return text

def remove_backslash_n(text):
    n=re.compile(r'\n')
    text = n.sub(r' ',text)
    return text

def custom_regex(text):
    ref=re.compile(r'ref|\.')
    ares=re.compile(r'ares|\(20(?:00|1[09]|2[01])\)')
    com=re.compile(r'com\([0-9]{1,4}\)')
    bignum = re.compile(r'[0-9]{5,30}')
    parenth = re.compile(r'\([0-9]{1,4}\)')
    email = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)')
    listing = re.compile(r'\([1-3]\)|\[[1-3]\]|[1-3]\.')
    page1 = re.compile(r'Page [0-9]{1,2}(?: \/ |\/)[0-9]{1,2}}')
    page2 = re.compile(r'Page [0-9]{1,2} of [0-9]{1,2}')
    page3 = re.compile(r'[0-9]{1,2} of [0-9]{1,2}')
    marks = re.compile(r'[\u2612\uF0B7\\x2D\u2022\u2713\x0c●©\x0f\x01\u2009\u20093▪]')
    questionaire = re.compile(r'1\-Not important at all|5\- Very important|2\- Not important|No opinion|4Important|3Neutral')
    single_letter = re.compile(r' [bcdefghjklmnopqrstuvwxyz] ')
    spaces = re.compile(r' +')
    text = ref.sub(r'',text)
    text = ares.sub(r'',text)
    text = com.sub(r'',text)
    text = bignum.sub(r'',text)
    text = parenth.sub(r'',text)
    text = email.sub(r'',text)
    text = listing.sub(r'',text)
    text = page1.sub(r'',text)
    text = page2.sub(r'',text)
    text = page3.sub(r'',text)
    text = marks.sub(r'',text)
    text = questionaire.sub(r'',text)
    text = single_letter.sub(r'',text)
    text = spaces.sub(r' ',text)
    return text

def clean(df,text_col='Text'):
    """
    Desc:   Pipeline that runs all cleaning functions on DataFrame with texts as
            string
    Input:  df DataFrame we want to clean
            text_col column name we want to clean
    Output: df with cleaned text_col column
    """
    df[text_col]=df[text_col].apply(lambda x : my_lower(x))
    df[text_col]=df[text_col].apply(lambda x : remove_URL(x))
    df[text_col]=df[text_col].apply(lambda x : remove_html(x))
    df[text_col]=df[text_col].apply(lambda x : remove_punct(x))
    df[text_col]=df[text_col].apply(lambda x : custom_regex(x))
    return df

def customize_stop_words(del_words, en):
    """
    Desc:   Adds custom stopwrods to the original set
    Input:  del_words we want to join defalut en stopwords 
            en model for langugae from spacy (en model)
    Output: df with cleaned text_col column
    """
    for l in del_words:
      en.vocab[l].is_stop = True
      en.Defaults.stop_words.add(l)

def tokenize(df,en,text_col='content'):
    """
    Desc:   Performs Spacy en model tokenization on documents texts and converts 
            them into spacy.Doc objects.
    Input:  df DataFrame to tokenize
            en model for langugae from spacy (en model)
            text_col column name we want to tokenize         
    Output: docs list of Doc objects (lemmatized words)
    """
    # tqdm.pandas()
    # docs = df[text_col].swifter.apply(en)
    # swifter not working with pandas 2.0
    docs = df[text_col].apply(en)
    return docs

"""# EDA"""

def plot_len_dist(docs,log_scale=False):
    """
    Desc:   Plots distribution of docs lengths
    Input:  docs list of Doc objects 
            log_scale bool describing if plot should be in log scale        
    Output: histplot of docs lengths
    """
    doc_lens = docs.str.len()
    return(doc_lens.hist(log_y=log_scale))

def get_nouns(docs):
    """
    Desc:   Get nouns from docs list
    Input:  docs list of Doc objects        
    Output: List of nouns that are not stop words from all docs list.
    """
    nouns = [token.text
          for doc in docs
          for token in doc
          if (not token.is_stop and
              not token.is_punct and
              token.pos_ == "NOUN")]
    return nouns

def get_nouns_list(docs):
    """
    Desc:   Get nouns from docs list
    Input:  docs list of Doc objects        
    Output: List of nouns that are not stop words from all docs list.
    """
    nouns_list = list()
    for doc in docs:
      nouns = [token.text
              for token in doc
              if (not token.is_stop and
                  not token.is_punct and
                  token.pos_ == "NOUN")]
      nouns_list.append(nouns)
    return nouns_list

def plot_counts(count_obj, names, width=800, height=400):
    """
    Desc:   Plots counted occurances of object / word
    Input:  count_obj DataFrame with name and counted nuber of occurences
            names list of column names in count_obj DataFrame       
    Output: Plot
    """
    fig = px.bar(count_obj,orientation='h', y=names[0], x=names[1], width=width, height=height)
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_layout(bargap=0.30, font={'size':10})
    return fig

def count_texts(texts,colnames=['obj', 'count'],n_obs=30):
    obj_freq = Counter(texts)
    common_obj = obj_freq.most_common(n_obs)
    count_obj = pd.DataFrame(common_obj, columns=colnames)
    return count_obj

def lemmatized_word_cloud(docs,width=800,height=400):
    """
    Desc:   Creates word cloud of lemmas from docs list
    Input:  docs list of Doc objects 
            width, height parameters of image       
    Output: Plots word cloud
    """
    lemmas = docs.apply(lambda doc: [token.lemma_ for token in doc if not token.is_stop if not token.is_punct if token.is_alpha])
    word_counts = Counter(lemmas.sum())
    wc = WordCloud(width=width, height=height)
    wc.generate_from_frequencies(frequencies=word_counts)
    plt.figure(figsize=(18,14))
    plt.imshow(wc)
    return word_counts

def get_lemmas_list(docs):
    lemmas = docs.apply(lambda doc: [token.lemma_ for token in doc 
                                     if not token.is_stop 
                                     if not token.is_punct 
                                     if token.is_alpha])
    return(lemmas)

def get_entities(docs):
    """
    Desc:   Gets all entitites from docs list
    Input:  docs list of Doc objects       
    Output: list of entities
    """
    entities = [(ent.text, ent.label_)
          for doc in docs
            for ent in doc.ents]
    return entities

def get_entities_list(docs):
    """
    Desc:   Gets all entitites from docs list
    Input:  docs list of Doc objects       
    Output: list of entities
    """
    entities_list = list()
    for doc in docs:
      entities = [(ent.text, ent.label_)
              for ent in doc.ents]
      entities_list.append(entities)
    return entities_list
  

def unique(list1):
    unique_list = []
    for x in list1:
      if x not in unique_list:
        unique_list.append(x)
    return unique_list

def organisation_like_entitites(docs, uniq=False):
    """
    Desc:   Gets all organisation like entitites from docs list
    Input:  docs list of Doc objects
            uniq bool describing wheter we want to obtain unique entities only       
    Output: list of entities
    """
    entities = get_entities(docs)
    if uniq:
      entities = unique(entities)
    entities_df = pd.DataFrame(entities, columns =['entity','type'])
    ls = ["EVENT","GPE","LAW","NORP","PERSON","ORG"]
    proper_ets = entities_df[entities_df['type'].isin(ls)]
    return proper_ets

def organisation_like_entitites_list(docs, uniq=False):
    """
    Desc:   Gets all organisation like entitites list from docs list
    Input:  docs list of Doc objects
            uniq bool describing wheter we want to obtain unique entities only       
    Output: list of entities
    """
    entities_list = list()
    entities = get_entities_list(docs)
    for enitity in entities:
      if uniq:
        enitity = unique(enitity)
      entities_df = pd.DataFrame(enitity, columns =['entity','type'])
      ls = ["EVENT","GPE","LAW","NORP","PERSON","ORG"]
      proper_ets = entities_df[entities_df['type'].isin(ls)]
      proper_ets = proper_ets['entity'].to_list()
      entities_list.append(proper_ets)
    return entities_list

"""## Tfidf"""

def dummy_fun(doc):
    """
    Desc:   required for tfidf_table
    """
    return doc

def tfidf_table(texts_df,en,top=10):
    """
    Desc:   With the usage of TfidfVectorizer prepares tdidf DataFrame
    Input:  texts_df DataFrame with single column containing texts
            en model for langugae from spacy (en model) 
            top number of how many most important lemmas will be included (importance
            by tfidf)
    Output: DataFrame where columns are paired as term_X and score_X. This pair
            describes importance of lemmas for Xth document
    """
    stopwords = en.Defaults.stop_words

    vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, norm=None)
    #vectorizer = TfidfVectorizer(analyzer='word',tokenizer=dummy_fun,preprocessor=dummy_fun,token_pattern=None) 
    transformed_documents = vectorizer.fit_transform(texts_df)
    #transformed_documents = vectorizer.fit_transform(docs)

    transformed_documents_as_array = transformed_documents.toarray()
    output_filenames = [range(len(transformed_documents_as_array))]

    docs_as_dfs = []
    for counter, doc in enumerate(transformed_documents_as_array):
        tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
        one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)
        docs_as_dfs.append(one_doc_as_df)
    top_tfidf = docs_as_dfs[0][:top]
    for i in range(len(docs_as_dfs)-1):
      top_tfidf = pd.concat([top_tfidf, docs_as_dfs[i+1][:10]], axis=1)

    tfidf_names = []
    for i in range(len(docs_as_dfs)):
      tfidf_names.append("term_"+str(i))
      tfidf_names.append("score_"+str(i))
    top_tfidf.columns = tfidf_names

    return top_tfidf

def counts_tfidf(top_tfidf, num = 40):
    """
    Desc:   Counts in how many documents the single lemma was in top lemmas
    Input:  top_tfidf DataFrame created by tfidf_table()
    Output: list with counted occurences of the lemma in tfidfs top lemmas
    """
    terms_tfidf = top_tfidf.loc[:, ::2]
    terms_list = []
    for i in range(len(terms_tfidf.columns)):
      for j in range(len(terms_tfidf)):
        terms_list.append(terms_tfidf.iloc[j,i])
    terms_freq = Counter(terms_list)
    common_terms = terms_freq.most_common(num)
    count_terms = pd.DataFrame(common_terms, columns=['term', 'count'])
    return count_terms

"""## Ngrams"""

def get_top_ngram(texts_df, stopwords, n=None, m=None, num = 30):
    """
    Desc:   Prepares top ngrams after removing the stop words from text given in
            texts_df list of strings
    Input:  texts_df DataFrame with single column containing texts
            stopwords that won't be included in the ngrams
            n,m sets range of ngram, f.ex. n=2,m=4 says that ngrams with 2,3 or 4
            words inside them will be searched
    Output: list of most common ngrams in given range
    """  
    vec = CountVectorizer(stop_words = stopwords, ngram_range=(n, m)).fit(texts_df)
    bag_of_words = vec.transform(texts_df)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:num]
    
def plot_ngram(top_ngrams):
    """
    Desc:   Prepares barplot for top ngrams
    Input:  top_ngrams list of most common ngrams
    """
    x,y=map(list,zip(*top_ngrams))
    fig= pyplot.subplots(figsize=(15,15))
    sns.barplot(x=y,y=x)

"""## Noun Chunks"""

def get_noun_chunks(docs, stopwords = False):
    """
    Desc:   Prepares list of noun chunks as strings present in all docs
    Input:  docs list of Doc objects 
            stopwords bool describving if we want to remove stopwords or not
    Output: list of noun chunks as strings
    """
    noun_chunks = []
    for doc in docs:
      for chunk in doc.noun_chunks:
        if stopwords:
          stop = True
          for w in chunk:
            if w.is_stop:
              stop = False
          if stop:
            noun_chunks.append(chunk.text)
        else:
          noun_chunks.append(chunk.text)
    return noun_chunks

def get_noun_chunks_list(docs, stopwords = False):
    """
    Desc:   Prepares list of noun chunks per document as strings present in all docs
    Input:  docs list of Doc objects 
            stopwords bool describving if we want to remove stopwords or not
    Output: list of noun chunks as strings
    """
    noun_chunks_list = []
    for doc in docs:
      noun_chunks = []
      for chunk in doc.noun_chunks:
        if stopwords:
          stop = True
          for w in chunk:
            if w.is_stop:
              stop = False
          if stop:
            noun_chunks.append(chunk.text)
        else:
          noun_chunks.append(chunk.text)
      noun_chunks_list.append(noun_chunks)
    return noun_chunks_list


def get_chunks(docs, stopwords = False):
    """
    Desc:   Prepares list of noun chunks present in all docs
    Input:  docs list of Doc objects 
            stopwords bool describving if we want to remove stopwords or not
    Output: list of noun chunks
    """
    chunks = list()
    if stopwords:
      for doc in docs:
        ok_chunks = list()
        for chunk in doc.noun_chunks:
          stop = True
          for w in chunk:
            if w.is_stop:
              stop = False
          if stop:
            ok_chunks = ok_chunks + chunk
        chunks = chunks + list(ok_chunks)
    else:
      for doc in docs:
        chunks = chunks + list(doc.noun_chunks)
    return chunks

def top_chunk_parents(chunks,count_chunks,n_chunks=10,n_parents=5):
    """
    Desc:   Prepares list of noun chunks present in all docs
    Input:  chunks list of noun chunks
            count_chunks counted list of noun chunks as strings (from get_noun_chunks)
            n_chunks number of most occuring chunks we want to consider
            n_parents number of parents we are looking for each chunk
    Output: list of DataFrames with 3 columns: chunk, parent, count
    """
    chunk_parents = []
    for chunk in chunks:
      if chunk.text in (list(count_chunks["chunk"][:n_chunks])):
        chunk_parents.append((chunk.text,chunk.root.head.text))
    ch_p_df = pd.DataFrame(chunk_parents, columns =['chunk','parent'])
    count_ch_p = ch_p_df.value_counts()
    count_ch_p = count_ch_p.reset_index()
    count_ch_p.columns =["chunk","parent","count"]
    most_common_parents = []
    for el in list(count_chunks["chunk"][:n_chunks]):
      p = count_ch_p[count_ch_p["chunk"]== el][:n_parents]
      most_common_parents.append(p)
    return most_common_parents

def count_texts(texts,colnames=['obj', 'count'],n_obs=30):
    obj_freq = Counter(texts)
    common_obj = obj_freq.most_common(n_obs)
    count_obj = pd.DataFrame(common_obj, columns=colnames)
    return count_obj

def plot_chunks_parents(most_common_parents):
    """
    Desc:   Prepares visualization for top_chunk_parents function
    Input:  most_common_parents - output of top_chunk_parents function
    """
    figure(figsize=(20, 12), dpi=100)
    plt.figure(1)
    for i in range(9):
      plt.subplot(331+i)
      plt.bar(most_common_parents[i]["parent"], most_common_parents[i]["count"])
      plt.title(most_common_parents[i]["chunk"].iloc[0])

def chunk_frequency(docs, n_top_chunks = 10, stopwords = False):
    """
    Desc:   Prepares a DataFrame with 4 columns: Chunk,Count,ChunkFrequency and 
            Percent. ChunkFrequency informs us in how many documents given noun 
            chunk existed and Percent gives us the percentage of this event in terms
            of total number of documents
    Input:  docs list of Doc objects 
            n_top_chunks number of most occuring chunks we want to consider
            stopwords bool describving if we want to remove stopwords or not
    Output: DataFrame with 4 columns: Chunk,Count,ChunkFrequency and 
            Percent.
    """
    freq_chunk = []
    for doc in docs:
      n_chunks = []
      for chunk in doc.noun_chunks:
        if stopwords:
          stop = True
          for w in chunk:
            if w.is_stop:
              stop = False
          if stop:
            n_chunks.append(chunk.text)
        else:
          n_chunks.append(chunk.text)
      freq_chunk.append(n_chunks)

    noun_chunks = get_noun_chunks(docs, stopwords)
    noun_chunks = list(filter(lambda x: len(x.split()) > 1, noun_chunks))
    count_chunks = count_texts(noun_chunks,['chunk', 'count'],n_top_chunks)
    common_chunks = count_chunks['chunk']
    count_chunks = count_chunks['count']

    count_chunk = []
    for ch in common_chunks:
      x=0
      for i in range(len(freq_chunk)):
        if ch in freq_chunk[i]:
          x+=1
      count_chunk.append(x)
    
    ch_name = []
    ch_count = []
    for ch in common_chunks:
      ch_name.append(ch)
    for ch in count_chunks:
      ch_count.append(ch)
    
    chunk_count_df = pd.DataFrame(ch_name)
    chunk_count_df["Count"] = ch_count
    chunk_count_df["CF"] = count_chunk
    chunk_count_df["Percent"] = [x / len(docs) for x in count_chunk]
    chunk_count_df.columns = ["Chunk","Count","ChunkFrequency", "Percent"]

    return chunk_count_df

def plot_count_chunks(chunk_count_df):
    """
    Desc:   Prepares a visualization for output of chunk_frequency() output
    Input:  DataFrame with 4 columns: Chunk,Count,ChunkFrequency and 
            Percent.
    """
    labels = list(chunk_count_df["Chunk"])

    x = np.arange(len(chunk_count_df["Chunk"]))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(20, 12))
    rects1 = ax.bar(x - width/2, chunk_count_df["Count"], width, label='Count')
    rects2 = ax.bar(x + width/2, chunk_count_df["ChunkFrequency"], width, label='ChunkFrequency')

    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()

    plt.show()

"""## Textrank"""

def textrank(docs, n_of_keyterms = 40):
    """
    Desc:   Performs Textrank and provides a visualization for it
    Input:  docs list of Doc objects
            n_of_keyterms number of keyterms returned by the analysis
    Output: Visualization of the textrank
    """
    keyterms = []
    for doc in docs:
      keyterms.append(textacy.extract.keyterms.textrank(doc))
    keyterms_list = []
    for i in range(len(docs)):
      keyterms_df = pd.DataFrame.from_dict(keyterms[i])
      for j in range(len(keyterms_df[0])):
        keyterms_list.append(keyterms_df[0][j])
    keyterms_freq = Counter(keyterms_list)
    common_keyterms = keyterms_freq.most_common(n_of_keyterms)
    count_keyterms = pd.DataFrame(common_keyterms, columns=['keyterm', 'count'])
    fig = px.bar(count_keyterms,orientation='h', y='keyterm', x='count')

    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_layout(bargap=0.30, font={'size':10})
    fig

"""## Sentiment"""

def count_sentiment(docs):
    """
    Desc:   Performs sentiment analysis involving poalrity and subjectibity measures
    Input:  docs list of Doc objects
    Output: DataFrame with 2 columns: polarity and subjectivity
    """
    sentiment_info = []
    for doc in docs:
      sentiment = (doc._.blob.polarity, doc._.blob.subjectivity)
      sentiment_info.append(sentiment)
    sentiment_df = pd.DataFrame.from_dict(sentiment_info)
    sentiment_df.columns=["polarity","subjectivity"]
    return sentiment_df