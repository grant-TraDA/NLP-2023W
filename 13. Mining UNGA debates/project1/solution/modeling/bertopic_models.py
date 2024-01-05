from bertopic import BERTopic
import os
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance


CORPUS_DIR = os.path.join("corpora", "UN General Debate Corpus", "TXT")
MODEL_DIR = "models"
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
    "distilbert": pipeline("feature-extraction", model="distilbert-base-cased"),
    "all-MiniLM-L12-v2": SentenceTransformer("all-MiniLM-L12-v2"),
    "roberta": pipeline("feature-extraction", model="roberta-base"),
}


def get_texts(path):
    all_texts = []

    # iterate through all UN Sessions
    for session in os.listdir(path):
        if session.startswith("."):
            continue

        # get the file for each country
        for file in os.listdir(os.path.join(path, session)):
            if file.startswith("."):
                continue

            with open(os.path.join(path, session, file), "r") as f:
                # get file text...
                text = f.read()

                # ...and metadata
                metadata = file.split(".txt")[0].split("_")
                context = {
                    "country": metadata[0],
                    "session": metadata[1],
                    "year": metadata[2],
                }
                all_texts.append((text, context))
    return all_texts


def create_and_save_model(texts, embedding_model, save_path):
    # create a sensible representation model
    representation_model = KeyBERTInspired()

    # create BERTopic instance with the provided embedding model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        representation_model=representation_model,
        top_n_words=15,
        calculate_probabilities=True,
        verbose=True,
    )
    # fit BERTopic to the provided texts
    topic_model.fit_transform([text[0] for text in texts])

    # save model at the provided path
    topic_model.save(
        save_path,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=embedding_model,
    )


def main():
    all_texts = get_texts(CORPUS_DIR)
    print("Loaded {} texts".format(len(all_texts)))

    # create and save all models
    for embedding_model_name, embedding_model in EMBEDDING_MODELS.items():
        print("Creating model for {}".format(embedding_model_name))
        save_path = os.path.join(MODEL_DIR, embedding_model_name)
        create_and_save_model(all_texts, embedding_model, save_path)


if __name__ == "__main__":
    main()
