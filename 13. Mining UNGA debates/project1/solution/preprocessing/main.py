metadata_path = "metadata/enhanced_metadata.csv"
corpora_path = "corpora/UN General Debate Corpus/TXT"

from text_preprocessing import get_processed_data

if __name__ == "__main__":
    get_processed_data(metadata_path, corpora_path)
