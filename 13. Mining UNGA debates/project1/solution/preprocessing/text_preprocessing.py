import os
from typing import List, Optional

from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy
from spacy.tokens import Doc, Token
import textdescriptives as td
from utils import read_txt, save_processed_data, read_processed_data


def load_metadata_and_texts(metadata_path: str, corpora_path: str) -> pd.DataFrame:
    """Merges the metadata and the texts into a single dataframe."""
    metadata_df = pd.read_csv(metadata_path)
    assert "text_path" in metadata_df.columns
    metadata_df["text"] = metadata_df["text_path"].apply(read_txt, corpora_path=corpora_path)
    return metadata_df


def get_filtered_tokens(spacy_text: Doc, additional_stop_words: Optional[List[str]]) -> List[Token]:
    """Processes the tokens in a document. Removes stop words, punctuation and non-alphabetic tokens."""
    return [
        token
        for token in spacy_text
        if not any([token.is_stop, token.is_punct, token.lemma_.lower() in additional_stop_words, not token.is_alpha])
    ]


def process_lemmas(doc: pd.Series) -> List[str]:
    """Makes tokens lemma lower case."""
    return [token.lemma_.lower() for token in doc]


def preprocess_text(
    df_with_texts: pd.DataFrame,
    spacy_model: str = "en_core_web_lg",
    n_process: int = 4,
    batch_size: int = 25,
    additional_stop_words: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Preprocesses the text in a dataframe using spacy."""
    nlp = spacy.load(spacy_model, exclude=["ner"])
    nlp.add_pipe("textdescriptives/all")
    df = df_with_texts.copy()

    res = pd.DataFrame()
    lemmas = []

    for doc in tqdm(nlp.pipe(df["text"].values, n_process=n_process, batch_size=batch_size), total=len(df)):
        tokens = get_filtered_tokens(doc, additional_stop_words=additional_stop_words)
        lemmas.append(process_lemmas(tokens))
        res_tmp = td.extract_df(doc, include_text=False)
        res = pd.concat([res, res_tmp], axis=0)

    df["lemmas"] = lemmas
    df_add = pd.concat(
        [df.reset_index(drop=True), res.reset_index(drop=True)],
        axis=1,
    )

    return df, df_add


def get_processed_data(
    metadata_path: Optional[str] = None,
    corpora_path: Optional[str] = None,
    additional_stop_words: List = [],
    spacy_model: str = "en_core_web_lg",
    processed_filename: str = "data_processed_add_features.pickle",
    n_process: int = 4,
    batch_size: int = 25,
) -> pd.DataFrame:
    """Returns the processed data. If possible, loads preprocessing results from a joblib file."""
    if os.path.isfile(processed_filename):
        print("Loading processed data")
        processed_data = read_processed_data(processed_filename)
    else:
        print("Preprocessing data using spacy pipeline")
        processed_data, processed_data_add_features = preprocess_text(
            load_metadata_and_texts(metadata_path, corpora_path),
            spacy_model,
            n_process,
            batch_size,
            additional_stop_words,
        )
        save_processed_data(processed_data, processed_data_add_features, processed_filename)
    return processed_data
