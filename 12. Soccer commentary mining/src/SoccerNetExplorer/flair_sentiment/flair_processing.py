import json
from copy import deepcopy
from pathlib import Path
from typing import List

from flair.data import Sentence
from flair.nn import Classifier
from tqdm import tqdm


def flair_sentence(segment: dict, tagger: Classifier) -> dict[str, float]:
    """
    Process a segment of text using Flair sentiment analysis.

    Args:
        segment (dict): A dictionary containing the text segment to analyze.
        tagger (Classifier): The Flair classifier used for sentiment analysis.

    Returns:
        dict: A dictionary containing the sentiment analysis results, including the sentiment score, tag, and scaled sentiment.
    """
    # Extract the text from the segment dictionary
    txt = segment["text"]

    # Create a Flair Sentence object
    sentence = Sentence(txt)

    # Predict the sentiment using the Flair classifier
    tagger.predict(sentence)

    # Calculate the scaled sentiment based on the sentiment tag
    sentiment_scaled = (
        (sentence.score - 0.5) * 2
        if sentence.tag == "POSITIVE"
        else -(sentence.score - 0.5) * 2
    )

    # Return the sentiment analysis results as a dictionary
    return {
        "sentiment": sentiment_scaled,
        "score": sentence.score,
        "tag": sentence.tag,
    }


def flair_file(pth: Path, tagger: Classifier) -> str:
    """
    Process a file with Flair sentiment analysis.

    Args:
        pth (Path): The path to the file.
        tagger (Classifier): The Flair sentiment classifier.

    Returns:
        str: A success message if the file was processed successfully, or an error message if the path is invalid.
    """
    # Check if the file path exists
    if not pth.exists():
        return "path error: " + str(pth)

    # Load the data from the file
    with open(pth, "r") as f:
        data = json.load(f)

    # Create a deep copy of the data
    data_new = deepcopy(data)

    # Process each segment in the data
    for segment in data_new["segments"]:
        # Perform Flair sentiment analysis on the segment
        segment["flair"] = flair_sentence(segment, tagger)

    # Save the updated data back to the file
    with open(pth, "w") as f:
        json.dump(data_new, f)

    # Return a success message
    return "success: " + str(pth)


def flair_sentiment(pths: List[Path], n_workers: int = 4) -> List[str]:
    """
    Extracts sentiment using Flair library for a list of file paths.

    Args:
        pths (List[Path]): List of file paths.
        n_workers (int, optional): Number of workers for parallel processing. Defaults to 4.

    Returns:
        List: List of sentiment results for each file path.
    """
    # Initialize an empty list to store the results
    results = []

    # Load the Flair sentiment classifier
    tagger = Classifier.load("sentiment")

    # Process each file path in parallel using multiple workers
    for pth in tqdm(pths, desc="Extracting flair sentiment"):
        # Perform Flair sentiment analysis on the file
        results.append(flair_file(pth, tagger))

    # Return the list of sentiment results
    return results
