from typing import Literal
from data_structures import SentimentAnnotation, AspectAnnotation, AspectBasedResults
from transforms import transform_aspects
import numpy as np
import pandas as pd
import openai
import spacy
from sentistrength import PySentiStr
from itertools import chain
from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints


class AspectAnotator:
    def __init__(
        self,
        stage1_tool: Literal["spacy", "chat_gpt"] = None,
        stage2_tool: Literal["sentistrength", "chat_gpt"] = None,
        full_tool: Literal["pyabsa"] = None,
        ss_jar_path: str = None,
        ss_lang_path: str = None,
    ) -> None:
        if stage1_tool is None and stage2_tool is None and (full_tool is None):
            raise ValueError("Select tool used for sentiment analysis!")

        if (stage1_tool is not None and stage2_tool is None) or (
            stage1_tool is None and stage2_tool is not None
        ):
            raise ValueError("Select two tools in two-step approach!")

        if (
            stage1_tool is not None or stage2_tool is not None
        ) and full_tool is not None:
            raise ValueError("Select one step approach or two steps approach!")

        self.stage1_tool = stage1_tool
        self.stage2_tool = stage2_tool
        self.full_tool = full_tool
        if stage1_tool == "spacy":
            # done here to avoid loading it with every annotation exercise
            self.stage1_annotator = spacy.load("en_core_web_sm")

        if stage2_tool == "sentistrength":
            if ss_jar_path == None:
                raise ValueError(
                    "Missing path for SentiStrength .jar! Note: Provide absolute path instead of relative path"
                )
            if ss_jar_path == None:
                raise ValueError(
                    "Missing path for SentiStrength data folder! Note: Provide absolute path instead of relative path"
                )
            self.stage2_annotator = PySentiStr()
            self.stage2_annotator.setSentiStrengthPath(ss_jar_path)
            self.stage2_annotator.setSentiStrengthLanguageFolderPath(ss_lang_path)

        if full_tool == "pyabsa":
            checkpoint_map = available_checkpoints()

            self.full_tool_annotator = ATEPC.AspectExtractor(
                "multilingual",
                auto_device=True,  # False means load model on CPU
                cal_perplexity=True,
            )

        # I'd rather not to leave it in here at least for now
        openai.api_key = ""

    def annotate(self, texts: list[str]) -> list[AspectAnnotation]:
        if isinstance(texts, str):
            texts = [texts]

        annotations = []
        aspects = []

        if self.stage1_tool == "spacy":
            for text in texts:
                doc = self.stage1_annotator(text)
                sentence = next(doc.sents)
                subjects = [
                    word
                    for word in sentence
                    if word.dep_ in ["nsubj"] and word.orth_ not in ["I", "you"]
                ]
                aspects.append(subjects)

        if self.stage2_tool == "sentistrength":
            rep = [len(a) for a in aspects]
            texts_rep = np.repeat(texts, rep)
            aspects_unlist = [a.orth_ for a in list(chain.from_iterable(aspects))]
            sentiments = self.stage2_annotator.getSentiment(
                texts_rep, keywords=aspects_unlist, score="trinary"
            )
            sentiments = [np.argmax(np.abs(sentiment)) for sentiment in sentiments]
            sentiments = list(map(self.map_senti, sentiments))
            sentiments = [
                SentimentAnnotation(text=aspects_unlist[i], label=sentiments[i])
                for i in range(len(sentiments))
            ]

            rep_cum = np.insert(np.cumsum(rep), 0, 0, axis=0)
            sentiments_grouped = [
                sentiments[rep_cum[i] : rep_cum[i + 1]] for i in range(len(rep_cum) - 1)
            ]
            annotations = [
                AspectAnnotation(text=texts[i], aspects=sentiments_grouped[i])
                for i in range(len(sentiments_grouped))
            ]

        if self.full_tool == "pyabsa":
            tool_annotations = self.full_tool_annotator.predict(
                texts,
                save_result=False,
                print_result=False,  # print the result
                ignore_error=True,  # ignore the error when the model cannot predict the input
            )

            annotations = [
                AspectAnnotation(
                    text=result["sentence"],
                    aspects=[
                        SentimentAnnotation(
                            text=aspect, label=sentiment, score=confidence
                        )
                        for aspect, sentiment, confidence in zip(
                            result["aspect"],
                            result["sentiment"],
                            result["confidence"],
                        )
                    ],
                )
                for result in tool_annotations
            ]

        return annotations

    def test_annotator(
        self,
        true_annotations: list[AspectAnnotation] | pd.DataFrame,
        id_column: str = None,
        text_column: str = None,
        aspect_column: str = None,
        sentiment_column: str = None,
    ) -> AspectBasedResults:
        if isinstance(true_annotations, pd.DataFrame):
            for col in [id_column, text_column, aspect_column, sentiment_column]:
                if col is None:
                    raise ValueError(f"Specify {col} if data frame is provided!")
            true_annotations = transform_aspects(
                true_annotations,
                id_column,
                text_column,
                aspect_column,
                sentiment_column,
            )
        texts = [aa.text for aa in true_annotations]
        predicted_annotations = self.annotate(texts)
        results = self.calculate_results(true_annotations, predicted_annotations)
        return results

    @staticmethod
    def calculate_results(
        true_annotations: list[AspectAnnotation],
        predicted_annotations: list[AspectAnnotation],
    ) -> AspectBasedResults:
        COR = 0
        INC = 0
        PAR = 0
        SPU = 0

        if isinstance(true_annotations, pd.DataFrame):
            true_annotations = transform_aspects(true_annotations)

        for pred_an, true_an in zip(predicted_annotations, true_annotations):
            for pred in pred_an.aspects:
                pred_aspect = pred.text
                pred_sentiment = pred.label.lower()
                matching_aspect = False
                for true_ in true_an.aspects:
                    true_aspect = true_.text
                    true_sentiment = true_.label.lower()
                    if pred_aspect == true_aspect:
                        if pred_sentiment == true_sentiment:
                            COR += 1
                        else:
                            INC += 1
                        matching_aspect = True
                        break
                if not matching_aspect:
                    for true_ in true_an.aspects:
                        true_aspect = true_.text
                        true_sentiment = true_.label.lower()
                        if true_aspect in pred_aspect or pred_aspect in true_aspect:
                            if pred_sentiment == true_sentiment:
                                PAR += 1
                            else:
                                INC += 1
                            matching_aspect = True
                            break
                if not matching_aspect:
                    SPU += 1

        MIS = np.sum([len(ta.aspects) for ta in true_annotations]) - COR - INC - PAR
        POS = COR + INC + PAR + MIS
        ACT = COR + INC + PAR + SPU

        return AspectBasedResults(
            correct=COR, incorrect=INC, missing=MIS, partial=PAR, spurious=SPU
        )

    @staticmethod
    def map_senti(sentiment):
        if sentiment == 0:
            return "positive"
        if sentiment == 1:
            return "negative"
        if sentiment == 2:
            return "neutral"
