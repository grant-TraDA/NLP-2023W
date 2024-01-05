import logging
import sys
from itertools import product
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.models.LDA import LDA
from tqdm import tqdm

sys.path.append("./preprocessing/")

from text_preprocessing import load_metadata_and_texts

COHERENCE_MEASURE = "c_v"
CORPUS_PATH = "./corpora/UN General Debate Corpus/TXT"
METADATA_PATH = "./metadata/enhanced_metadata.csv"
MODEL_NAMES = [
    "LDA",
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "distilbert",
    "roberta",
]
MODELS_DIR = "./models/"
NUM_TOPICS_VALUES = [10, 20, 50]
OCTIS_DATASET_PATH = "./metrics/octis_dataset/"
OUTPUT_PATH = "./metrics/metric_values.csv"


class DisableLogging:
    def __init__(self, level):
        self._level = level

    def __enter__(self):
        logging.disable(self._level)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


dataset = Dataset()
dataset.load_custom_dataset_from_folder(OCTIS_DATASET_PATH)
corpus = dataset.get_corpus()

metric_values = []
for model_name, num_topics in tqdm(list(product(MODEL_NAMES, NUM_TOPICS_VALUES))):
    if model_name == "LDA":
        model = LDA(num_topics)
        output = model.train_model(dataset)
    else:
        with DisableLogging(logging.ERROR):
            model = BERTopic.load(Path(MODELS_DIR) / model_name)
        output = {
            "topics": [
                [word for word, _ in topic]
                for topic in list(model.get_topics().values())[1 : num_topics + 1]
            ]
        }

    coherence_metric = Coherence(corpus, measure=COHERENCE_MEASURE, processes=-1)
    coherence_value = coherence_metric.score(output)
    metric_values.append(
        {
            "model_name": model_name,
            "num_topics": num_topics,
            "metric_name": "coherence",
            "metric_value": coherence_value,
        }
    )

    diversity_metric = TopicDiversity()
    diversity_value = diversity_metric.score(output)
    metric_values.append(
        {
            "model_name": model_name,
            "num_topics": num_topics,
            "metric_name": "diversity",
            "metric_value": diversity_value,
        }
    )

    pd.DataFrame(metric_values).to_csv(OUTPUT_PATH, index=False)
