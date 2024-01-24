import os
from pathlib import Path
from typing import List, Dict, Any

import librosa
import numpy as np
import pandas as pd
from loguru import logger
from speechbrain.pretrained.interfaces import foreign_class
from torch import Tensor
from tqdm import tqdm

from utils import configure_logger


def find_audio_files(directory: str) -> List[str]:
    """
    Scan a directory recursively to find all MP3 audio files.

    Parameters
    ----------
    directory : str
        The directory path to search for MP3 files.

    Returns
    -------
    List[str]
        A list of paths to the found MP3 files.
    """
    logger.info(f"Searching for MP3 files in {directory}")
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp3"):
                full_path = os.path.join(root, file)
                audio_files.append(full_path)
    logger.debug(f"Found {len(audio_files)} MP3 files.")
    return audio_files


def process_audio_file(file_path: str, classifier: Any, batch_duration: int = 120, extract_duration: int = 10) -> List[Dict[str, Any]]:
    """
    Processes an audio file, splits it into extracts, and returns predictions for each extract.

    Parameters
    ----------
    file_path : str
        The file path of the audio file to be processed.
    classifier : Any
        The classifier object used for audio file classification.
    batch_duration : int, optional
        Duration of each batch in seconds, by default 120.
    extract_duration : int, optional
        Duration of each extract in seconds, by default 10.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing the classification results for each audio extract.
    """
    # Load audio
    audio_data, sr = librosa.load(file_path, sr=16_000)
    total_frames = len(audio_data)

    results = []
    for start in range(0, total_frames, int(batch_duration * sr)):
        end = start + int(batch_duration * sr)
        batch_audio = audio_data[start:end]

        # Split batch into extracts
        extracts = [batch_audio[i:i + int(extract_duration * sr)] for i in
                    range(0, len(batch_audio), int(extract_duration * sr))]
        extracts = [extract for extract in extracts if
                    len(extract) == int(extract_duration * sr)]
        if extracts:
            out_prob, score, index, text_lab = classifier.classify_batch(
                Tensor(np.array(extracts)))

            start_frames = list(range(start, end, int(extract_duration * sr)))
            end_frames = list(range(start + int(extract_duration * sr), end + int(extract_duration * sr), int(extract_duration * sr)))
            for i, extract in enumerate(extracts):
                results.append({
                    "start_frame": start_frames[i],
                    "stop_frame": end_frames[i],
                    "prob_neutral": out_prob.numpy()[i][0],
                    "prob_angry": out_prob.numpy()[i][1],
                    "prob_happy": out_prob.numpy()[i][2],
                    "prob_sad": out_prob.numpy()[i][3],
                    "score": score.numpy()[i],
                    "text_lab": text_lab[i],
                    "index": index.numpy()[i],
                })

    return results


def save_results_to_csv(results: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save the results to a CSV file.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        The classification results to save.
    file_path : str
        The file path for the CSV file to be saved.
    """
    df = pd.DataFrame(results)

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    configure_logger("emotion_recognition")

    # Initialize speech emotion detection model
    logger.info("Initializing classifier")
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )
    # temporary for a single file load
    # audio_files = ["./../anger.wav"]
    # audio_files = [
    #     "./../data/audio/france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG/1_224p.mp3"]
    directory = "./../data/audio"
    audio_files = find_audio_files(directory)
    for file in tqdm(audio_files):
        logger.debug(f"Extraction emotion from: {file}")
        results = process_audio_file(file, classifier)

        # Create the alternative path (change mp3 to csv) change "audio" to "emotion"
        file = Path(file)
        parts = list(file.parts)
        parts[parts.index('audio')] = 'emotion'
        modified_path = Path(*parts)
        csv_path = modified_path.with_suffix('.csv')

        # Create new directory if needed
        new_directory = csv_path.parent
        new_directory.mkdir(parents=True, exist_ok=True)

        # Save the model output
        save_results_to_csv(results, str(csv_path))
