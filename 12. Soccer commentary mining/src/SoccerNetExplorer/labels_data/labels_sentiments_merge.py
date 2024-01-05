import pandas as pd
import numpy as np
import json
import re
import tqdm


def merge_sentiments_labels(sentiments_path, labels_path, encoder_path, output_path):
    """
    Function merges dataframes with sentiments data and labels data
    (both loaded from dedicated .csv files) and then, saves it as
    a .csv file. Within the merging process, the encoding-decoding .json
    file is loaded in order to extracts the matches id which is the key in
    merging process.

    Parameters
    ----------
    sentiments_path: str - path to the .csv file with the sentiments data

    labels_path: str - path to the .csv file with the labels data

    encoder_path: str - path to the .json file storing the ids of the matches
        assigned in the earlier process of data processing

    output_path: Path - path to the output .csv

    """

    df_sentiments = pd.read_csv(sentiments_path).drop(columns='Unnamed: 0')
    df_labels = pd.read_csv(labels_path)
    labels_to_name = {current_label:re.sub(r"[ ->/]+",'_', current_label.lower()) 
                      for current_label in df_labels['label'].unique()}

    with open(encoder_path, "r") as f:
        match_encoder = json.load(f)

    df_labels['match_id'] = df_labels['game_video_to_encode'].map(match_encoder['encoding'])
    df_sentiments_labels = pd.DataFrame()

    # due to the probable memory limitations, the merging process is conducted within
    # batches - every batch is the specific match_id
    for id in tqdm(df_sentiments['match_id'].unique(), desc="Unpacking objects"):
        df_curr = df_sentiments.loc[df_sentiments['match_id']==id].reset_index().drop(columns='index')
        df_curr['middle_time'] = (df_curr['start'] + df_curr['end']) / 2

        # we do not consider match_id for which there are no labels available
        if df_labels.loc[(df_labels['match_id']==id)].shape[0] == 0:
            df_sentiments_labels = pd.concat([df_sentiments_labels,df_curr])
            continue

        for current_label in labels_to_name.keys():
            current_label_colname = labels_to_name[current_label]

            curr_times = df_labels.loc[(df_labels['match_id']==id) & (df_labels['label']==current_label)]['game_time_sec'].tolist()
            curr_times = np.array([-np.inf] + curr_times + [np.inf])

            df_curr[f'{current_label_colname}_prev_event_time'] = curr_times[np.searchsorted(curr_times, df_curr['middle_time'])-1]
            df_curr[f'{current_label_colname}_next_event_time'] = curr_times[np.searchsorted(curr_times, df_curr['middle_time'])]

            df_curr[f'time_from_prev_{current_label_colname}'.lower()] = df_curr['middle_time'] - df_curr[f'{current_label_colname}_prev_event_time']
            df_curr[f'time_to_next_{current_label_colname}'.lower()] = df_curr[f'{current_label_colname}_next_event_time'] - df_curr['middle_time']

            df_curr[f'{current_label_colname}_flag'.lower()] = ((df_curr['start'] <= df_curr[f'{current_label_colname}_prev_event_time'.lower()])
                                                | (df_curr['end'] >= df_curr[f'{current_label_colname}_next_event_time'.lower()])).astype(int)

        df_sentiments_labels = pd.concat([df_sentiments_labels,df_curr])

    df_sentiments_labels.to_csv(output_path, index=None)
    
    return
