import json
from copy import deepcopy
from pathlib import Path
from typing import List

from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def vader_sentence(
    segment: dict, analyzer: SentimentIntensityAnalyzer
) -> dict[str, float]:
    """
    Calculate the sentiment scores for a given text segment using the VADER sentiment analysis tool.

    Args:
        segment (dict): A dictionary containing the text segment to be analyzed.
        analyzer (SentimentIntensityAnalyzer): An instance of the VADER sentiment analyzer.

    Returns:
        dict: A dictionary containing the sentiment scores for the text segment, including positive, negative, neutral, and compound scores.
    """
    # Extract the text from the segment
    txt = segment["text"]

    # Calculate the sentiment scores using VADER
    scores = analyzer.polarity_scores(txt)

    # Create a dictionary to store the sentiment scores
    tmp = {
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "compound": scores["compound"],
    }

    return tmp


def vader_file(pth: Path, analyzer: SentimentIntensityAnalyzer) -> str:
    """
    Process a file using VADER sentiment analysis.

    Args:
        pth (Path): The path to the file.
        analyzer (SentimentIntensityAnalyzer): The VADER sentiment analyzer.

    Returns:
        str: A success message if the file was processed successfully, or an error message if the path does not exist.
    """
    # Check if the path exists
    if not pth.exists():
        return "path error: " + str(pth)

    # Load the data from the file
    with open(pth, "r") as f:
        data = json.load(f)

    # Create a deep copy of the data
    data_new = deepcopy(data)

    # Process each segment in the data
    for segment in data_new["segments"]:
        segment["vader"] = vader_sentence(segment, analyzer)

    # Save the updated data back to the file
    with open(pth, "w") as f:
        json.dump(data_new, f)

    return "success: " + str(pth)


def vader_sentiment(pths: List[Path], n_workers: int = 4) -> List[str]:
    """
    Calculate the VADER sentiment scores for a list of files.

    Args:
        pths (List[Path]): A list of file paths.
        n_workers (int, optional): The number of workers to use for parallel processing. Defaults to 4.

    Returns:
        List: A list of sentiment scores for each file.
    """
    results = []
    analyzer = SentimentIntensityAnalyzer()
    for pth in tqdm(pths, desc="Extracting vader sentiment"):
        results.append(vader_file(pth, analyzer))
    return results
