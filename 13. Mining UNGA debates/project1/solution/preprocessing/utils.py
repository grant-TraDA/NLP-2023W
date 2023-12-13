import os
import pickle
import pandas as pd


def read_txt(path: str, corpora_path: str) -> str:
    """Reads a text file and returns the first line."""
    with open(os.path.join(corpora_path, path)) as f:
        text = f.read()
    return text if len(text) > 0 else ""


def save_processed_data(processed_data: pd.DataFrame, processed_data_add_features: pd.DataFrame, processed_path: str):
    """Saves the processed data to a pickle file."""
    with open(processed_path, "wb") as f:
        pickle.dump(processed_data, f)
    with open(processed_path.replace(".pickle", "_add_features.pickle"), "wb") as f:
        pickle.dump(processed_data_add_features, f)


def read_processed_data(processed_path: str) -> pd.DataFrame:
    """Reads the processed data from a pickle file."""
    with open(processed_path, "rb") as f:
        processed_data = pickle.load(f)
    return processed_data
