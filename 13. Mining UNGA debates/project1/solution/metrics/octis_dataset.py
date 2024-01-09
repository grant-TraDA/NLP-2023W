import sys
from pathlib import Path

sys.path.append("./preprocessing/")

from utils import read_processed_data

INPUT_PATH = "./data_processed_add_features.pickle"
OUTPUT_PATH = "./metrics/octis_dataset/"

if __name__ == "__main__":
    output_path = Path(OUTPUT_PATH)
    if not output_path.is_dir():
        output_path.mkdir()
    
    corpus = read_processed_data(INPUT_PATH)["lemmas"]
    
    with (output_path / "corpus.tsv").open("w") as f:
        for document in corpus:
            f.write(" ".join(document) + "\t" + "train" + "\n")
