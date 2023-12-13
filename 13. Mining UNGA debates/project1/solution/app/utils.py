from collections import Counter
import pandas as pd
import os

PATH_TO_DATA = "../data_processed_add_features.pickle"
PATH_TO_MODEL = "../models/all-mpnet-base-v2"
PATH_TO_CORPUS = "../corpora/UN General Debate Corpus/TXT"
plotly_config = {"displayModeBar": False}

text_descriptive_attributes_mapping = {
    "Number of tokens": "n_tokens",
    "Number of unique tokens": "n_unique_tokens",
    "Fraction of unique tokens": "proportion_unique_tokens",
    "Number of characters": "n_characters",
    "Number of sentences": "n_sentences",
    "Token length - mean": "token_length_mean",
    "Token length - median": "token_length_median",
    "Token length - standard deviation": "token_length_std",
    "Sentence length - mean": "sentence_length_mean",
    "Sentence length - median": "sentence_length_median",
    "Sentence length - standard deviation": "sentence_length_std",
    "Readability - Flesch Reading Ease": "flesch_reading_ease",
    "Readability - Flesch-Kincaid Grade": "flesch_kincaid_grade",
    "Readability - SMOG": "smog",
    "Readability - Gunning-Fog": "gunning_fog",
    "Readability - Automated Readability Index": "automated_readability_index",
    "Readability - Coleman-Liau Index": "coleman_liau_index",
    "Readability - LIX": "lix",
    "Readability - RIX": "rix",
    "Proportion of nouns": "pos_prop_NOUN",
    "Proportion of verbs": "pos_prop_VERB",
    "Proportion of adjectives": "pos_prop_ADJ",
    "Proportion of adverbs": "pos_prop_ADV",
    "Proportion of pronouns": "pos_prop_PRON",
    "Coherence (first-order)": "first_order_coherence",
    "Coherence (second-order)": "second_order_coherence",
}

metadata_vars_mapping = {
    "Population": "Population",
    "Total Fertility Rate": "TFR",
    "Human Development Index": "HDI",
    "GDP (constant 2015 US$)": "GDP",
    "Unemployment Rate": "Unemployment Rate",
    "Gini Index": "Gini",
    "CO2 Emissions per Capita [t]": "CO2",
    "Democracy Index": "Democracy Index",
}


def count_lemmas(lemmas, max_words=15, additional_words=[]):
    lemmas = [word for word in lemmas if word not in additional_words]
    return Counter(lemmas).most_common(max_words)


def prepare_lemmas_counter_dataframe(counter_list):
    return pd.DataFrame(counter_list, columns=["word", "count"]).sort_values(
        by="count", ascending=False
    )


def get_texts(path):
    all_texts = []

    # iterate through all UN Sessions
    for session in os.listdir(path):
        if session.startswith("."):
            continue

        # get the file for each country
        for file in os.listdir(os.path.join(path, session)):
            if file.startswith("."):
                continue

            with open(os.path.join(path, session, file), "r") as f:
                # get file text...
                text = f.read()

                # ...and metadata
                metadata = file.split(".txt")[0].split("_")
                context = {
                    "country": metadata[0],
                    "session": metadata[1],
                    "year": metadata[2],
                }
                all_texts.append((text, context))
    return all_texts
