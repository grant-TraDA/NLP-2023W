import ast
import os

import pandas as pd


def read_news_data_to_dataframe(path):
    df = pd.read_json(path)
    return df


def read_all_news_in_dir(dir):
    filenames = os.listdir(dir)
    df = pd.DataFrame()
    for filename in filenames:
        df = pd.concat(
            [df, read_news_data_to_dataframe(dir + filename)], ignore_index=True
        )
    return df


def combine_lede_and_text(df):
    df[["lede", "text"]] = df[["lede", "text"]].fillna("")
    for i, row in df.iterrows():
        df.loc[i, "whole_text"] = row.lede + row.text
    return df


def remove_text_formatting(df):
    for i, row in df.iterrows():
        df.loc[i, "whole_text"] = (
            row.whole_text.replace("<b>", "")
            .replace("</b>", "")
            .replace("\n\n", " ")
            .replace("\n", " ")
        )
    return df


def correct_literals(dataframe, columns=["keywords", "categories"]):
    for column in columns:
        for i, row in dataframe.iterrows():
            new_value = ast.literal_eval(row[column])
            new_value = [x.strip() for x in new_value]
            dataframe.at[i, column] = new_value
    return dataframe


def convert_to_only_best_sentiment(
    dataframe, columns=["keywords_sentiment", "ner_sentiment"]
):
    for column in columns:
        for i, row in dataframe.iterrows():
            if pd.isnull(row[column]):
                dataframe.at[i, column] = dict()
                continue
            sentiment_dict = ast.literal_eval(row[column])[0]
            if len(sentiment_dict) > 0:
                sentiment_dict = {
                    key: find_label_with_highest_score(val)
                    for key, val in sentiment_dict.items()
                }
            dataframe.at[i, column] = sentiment_dict
    return dataframe


def find_label_with_highest_score(nested_list):
    max_score = 0
    best_idx = -1
    nested_list = nested_list[0]
    for label_idx in range(len(nested_list)):
        if nested_list[label_idx]["score"] > max_score:
            max_score, best_idx = nested_list[label_idx]["score"], label_idx
    return nested_list[best_idx]["label"]
