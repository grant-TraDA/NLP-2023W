import numpy as np
from typing import Literal
from data_structures import SentimentAnnotation, OrdinaryResults
from flair.data import Sentence
from flair.nn import Classifier
from sentistrength import PySentiStr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import openai


class OverallAnotator:
    def __init__(
        self,
        tool: Literal["chat_gpt", "flair", "sentistrength"],
        ss_jar_path: str = None,
        ss_lang_path: str = None,
        gpt_key: str = None,
    ) -> None:
        self.tool = tool
        if tool == "flair":
            # done here to avoid loading it with every annotation exercise
            self.tagger = Classifier.load("sentiment")
        if tool == "sentistrength":
            self.tagger = PySentiStr()
            if ss_jar_path == None:
                raise ValueError(
                    "Missing path for SentiStrength .jar! Note: Provide absolute path instead of relative path"
                )
            if ss_jar_path == None:
                raise ValueError(
                    "Missing path for SentiStrength data folder! Note: Provide absolute path instead of relative path"
                )
            self.tagger.setSentiStrengthPath(ss_jar_path)
            self.tagger.setSentiStrengthLanguageFolderPath(ss_lang_path)

        openai.api_key = gpt_key

    def annotate(self, texts: list[str]) -> list[SentimentAnnotation]:
        if isinstance(texts, str):
            texts = [texts]

        annotations = []

        if self.tool == "flair":
            for text in texts:
                sentence = Sentence(text)
                self.tagger.predict(sentence)
                annotations.append(
                    SentimentAnnotation(
                        text=text, label=sentence.tag, score=sentence.score
                    )
                )

        if self.tool == "chat_gpt":
            for text in texts:
                message = [
                    {
                        "role": "system",
                        "content": f"""For text below provide me a sentiment analysis label and score in the format:
                                        Label: <label you suggest>
                                        Score: <score you suggest>
                                        
                                        Text:
                                        {text}""",
                    }
                ]
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=message
                )
                reply = chat.choices[0].message.content

                try:
                    label, score = reply.split("\n")
                    label = label.removeprefix("Label: ")
                    score = score.removeprefix("Score: ")
                    score = float(score)
                except:
                    ValueError(f"Something wrong in the response: {reply}!")

                annotations.append(
                    SentimentAnnotation(text=text, label=label, score=score)
                )
        if self.tool == "sentistrength":
            for text in texts:
                sentiment_score = self.tagger.getSentiment(text, score="trinary")
                sentiment_index = [
                    np.argmax(np.abs(sentiment)) for sentiment in sentiment_score
                ]
                sentiment = list(map(self.map_senti, sentiment_index))[0]
                annotations.append(
                    SentimentAnnotation(
                        text=text,
                        label=sentiment,
                        score=sentiment_score[0][sentiment_index[0]],
                    )
                )

        return annotations

    def test_annotator(self, texts, true_labels) -> OrdinaryResults:
        predicted_labels = self.annotate(texts)
        predicted_labels = [
            predicted_label.label.lower() for predicted_label in predicted_labels
        ]
        true_labels = [true_label.lower() for true_label in true_labels]
        results = self.calculate_results(predicted_labels, true_labels)
        return results

    def calculate_results(self, true_labels, predicted_labels) -> OrdinaryResults:
        if len(true_labels) != len(predicted_labels):
            raise ValueError(
                "Lenghts of true_labels and predicted_labels must be equal!"
            )

        return OrdinaryResults(
            global_accuracy=accuracy_score(y_true=true_labels, y_pred=predicted_labels),
            macro_precision=precision_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="macro",
                zero_division=0,
            ),
            macro_recall=recall_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="macro",
                zero_division=0,
            ),
            macro_f1=f1_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="macro",
                zero_division=0,
            ),
            micro_precision=precision_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="micro",
                zero_division=0,
            ),
            micro_recall=recall_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="micro",
                zero_division=0,
            ),
            micro_f1=f1_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="micro",
                zero_division=0,
            ),
        )

    @staticmethod
    def map_senti(sentiment):
        if sentiment == 0:
            return "positive"
        if sentiment == 1:
            return "negative"
        if sentiment == 2:
            return "neutral"
