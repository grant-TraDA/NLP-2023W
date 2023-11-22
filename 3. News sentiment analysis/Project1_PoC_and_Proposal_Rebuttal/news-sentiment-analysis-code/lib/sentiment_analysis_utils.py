import pandas as pd
import os


def read_news_data_to_dataframe(path):
    df = pd.read_json(path)

    return df


def read_all_news_in_dir(dir):
    filenames = os.listdir(dir)
    df = pd.DataFrame()
    for filename in filenames:
        df = pd.concat([df, read_news_data_to_dataframe(dir + filename)], ignore_index=True)

    return df


def combine_lede_and_text(df):
    df[['lede', 'text']] = df[['lede', 'text']].fillna('')

    for i, row in df.iterrows():
        df.loc[i, 'whole_text'] = row.lede + row.text

    return df


def remove_text_formatting(df):
    for i, row in df.iterrows():
        df.loc[i, 'whole_text'] = row.whole_text.replace('<b>', '').replace('</b>', '').replace('\n\n', ' ').replace(
            '\n', ' ')

    return df
