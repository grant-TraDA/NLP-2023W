import sys
from pathlib import Path

sys.path.append("./preprocessing/")

from text_preprocessing import get_processed_data

OUTPUT_PATH = "./metrics/octis_dataset/"

if __name__ == "__main__":
    output_path = Path(OUTPUT_PATH)
    if not output_path.is_dir():
        output_path.mkdir()
    
    corpus = get_processed_data()["lemmas"]
    
    with (output_path / "corpus.tsv").open("w") as f:
        for document in corpus:
            f.write(" ".join(document) + "\t" + "train" + "\n")
