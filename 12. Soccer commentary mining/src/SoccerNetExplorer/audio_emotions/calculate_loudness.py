from pathlib import Path

from loguru import logger
from pydub import AudioSegment
from tqdm import tqdm

from SoccerNetExplorer.audio_emotions.emotion_recognition import (
    save_results_to_csv,
)
from SoccerNetExplorer.audio_emotions.utils import configure_logger


def find_mp3_files(directory: Path):
    """
    Find all MP3 files in the specified directory.

    Parameters
    ----------
    directory : Path
        Directory to search for MP3 files.

    Returns
    -------
    list of Path
        List of paths to MP3 files.
    """
    return list(directory.rglob("*.mp3"))


def calculate_loudness(file_path, interval_samples=16_000, sample_rate=16000):
    """
    Analyzes the audio file in intervals based on the number of raw samples to determine the average decibels.

    Parameters
    ----------
    file_path : Path
        Path to the audio file.
    interval_samples : int, optional
        Number of samples per interval (default is 16000 samples).
    sample_rate : int, optional
        Sample rate to be used for audio analysis (default is 16000 Hz).

    Returns
    -------
    list of float
        List of average decibels for each interval over a single audio file.
    """
    audio = AudioSegment.from_file(str(file_path)).set_frame_rate(sample_rate)
    interval_results = []
    for start_samples in range(0, len(audio), interval_samples):
        end_samples = start_samples + interval_samples
        chunk = audio[start_samples:end_samples]
        result = {
            "start_frame": start_samples,
            "end_frame": end_samples,
            "dBFS": chunk.dBFS,
        }
        interval_results.append(result)
    return interval_results


def loudness(
    base_audio: Path,
    base_loudness: Path,
    log_pth: Path,
):
    configure_logger("loudness", log_pth)
    base_directory = base_audio  # Base directory containing audio files
    output_directory = base_loudness  # Directory to save the CSV file
    output_directory.mkdir(
        parents=True, exist_ok=True
    )  # Ensure output directory exists

    audio_files = find_mp3_files(base_directory)
    for file in tqdm(audio_files):
        logger.debug(f"Extraction emotion from: {file}")
        results = calculate_loudness(file)

        # Create the alternative path (change mp3 to csv) change "audio" to "emotion"
        file = Path(file)
        parts = list(file.parts)
        parts[parts.index("audio")] = "loudness"
        modified_path = Path(*parts)
        csv_path = modified_path.with_suffix(".csv")

        # Create new directory if needed
        new_directory = csv_path.parent
        new_directory.mkdir(parents=True, exist_ok=True)

        # Save the output
        save_results_to_csv(results, str(csv_path))


if __name__ == "__main__":
    configure_logger("loudness")
    base_directory = Path(
        "./../data/audio/france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG"
    )  # Base directory containing audio files
    output_directory = Path(
        "./../data/loudness"
    )  # Directory to save the CSV file
    output_directory.mkdir(
        parents=True, exist_ok=True
    )  # Ensure output directory exists

    audio_files = find_mp3_files(base_directory)
    for file in tqdm(audio_files):
        logger.debug(f"Extraction emotion from: {file}")
        results = calculate_loudness(file)

        # Create the alternative path (change mp3 to csv) change "audio" to "emotion"
        file = Path(file)
        parts = list(file.parts)
        parts[parts.index("audio")] = "loudness"
        modified_path = Path(*parts)
        csv_path = modified_path.with_suffix(".csv")

        # Create new directory if needed
        new_directory = csv_path.parent
        new_directory.mkdir(parents=True, exist_ok=True)

        # Save the output
        save_results_to_csv(results, str(csv_path))
