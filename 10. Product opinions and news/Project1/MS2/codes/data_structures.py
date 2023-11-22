from dataclasses import dataclass, field
from typing import Literal, Optional, Annotated
import plotnine as p9
from plotnine import ggplot, aes
import pandas as pd
import numpy as np


@dataclass
class SentimentAnnotation:
    """
    Contains information about single sentiment annotation.

    Parameters
    ----------

    text : string
        Contain text that is labeled.
    label : string
        Label of the annotation e.g. 'positive', 'negative' and etc.
    score: float
        Score of the annotation, the more, the better. Optional since not all tools returns that.
    """

    text: str
    label: str
    score: Optional[float] = None


@dataclass
class AspectAnnotation:
    """
    Contains information about aspect sentiment annotation.

    Parameters
    ----------

    text : string
        Contains text that is labeled.
    aspects: list[SentimentAnnotation]
        Contains list of annotations for each aspect.
    score: float
        Score of the annotation, the more, the better. Optional since not all tools returns that.
    """

    text: str
    aspects: list[SentimentAnnotation]


@dataclass
class OrdinaryResults:
    """
    Contains results for the "ordinary" annotation where only label is predicted

    Parameters
    ----------
    global_accuracy : float
        Googleit
    macro_precision : float
        Googleit
    macro_recall : float
        Googleit
    macro_f1 : float
        Googleit
    micro_precision : float
        Googleit
    micro_recall : float
        Googleit
    micro_f1 : float
        Googleit


    """

    global_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    # automation similar to class below can be implemented

    def plot(self):
        res_df = pd.DataFrame(
            {
                "names": [
                    "\u200bGlobal accuracy",
                    "\u200cMacro precision",
                    "\u200cMicro precision",
                    "\u200dMacro recall",
                    "\u200dMicro recall",
                    "\u200eMacro F1",
                    "\u200eMicro F1",
                ],
                "values": np.round(
                    np.array(
                        [
                            self.global_accuracy,
                            self.macro_precision,
                            self.micro_precision,
                            self.macro_recall,
                            self.micro_recall,
                            self.macro_f1,
                            self.micro_f1,
                        ]
                    ),
                    2,
                ),
            }
        )
        dodge_text = p9.position_dodge(width=0.9)
        ccolor = "black"
        bar_color = "#130069"

        p = (
            ggplot(res_df, aes(x="names", y="values"))
            + p9.geom_col(
                stat="identity", position="dodge", show_legend=False, fill=bar_color
            )
            + p9.lims(y=(0, 1))
            + p9.geom_text(
                aes(label="values"),
                position=dodge_text,
                size=11,
                va="bottom",
                format_string="{}",
            )
            + p9.theme(
                panel_background=p9.element_rect(fill="white"),
                axis_title_y=p9.element_blank(),
                axis_title_x=p9.element_blank(),
                axis_line_x=p9.element_line(color="black"),
                axis_line_y=p9.element_blank(),
                axis_text_y=p9.element_blank(),
                axis_text_x=p9.element_text(color=ccolor, rotation=90),
                axis_ticks_major_y=p9.element_blank(),
                panel_grid=p9.element_blank(),
                panel_border=p9.element_blank(),
            )
        )

        return p


@dataclass
class AspectBasedResults:
    """
    Contains results for the aspect-based annotation where model predict
    place of the annotation and the label

    Parameters
    ----------

    correct : int
        if the observation and its label is the same as the gold-standard annotation
    incorrect : int
        if the observation is the same as the gold-standard annotation, but has incorrect label
    partial : int
        if the observation partially overlaps the gold-standard annotation and has correct label
    missing : int
        if a gold-standard annotation does not occur in result dataset
    spurius : int
        if the observation does not occur in the gold-standard annotation
    possible : int
        the number of annotations in the gold-standard which contribute to the final score
    actual : int
        the total number of annotations produced by the system
    precision : float
        correct/actual
    recall : float
        correct/possible
    f1 : float
        defined as ordinary f1 based on the precision and recall values

    """

    correct: int
    incorrect: int
    partial: int
    missing: int
    spurious: int
    possible: int = field(init=False)
    actual: int = field(init=False)
    precision: float = field(init=False)
    recall: float = field(init=False)
    f1: float = field(init=False)

    def __post_init__(self):
        self.possible = self.correct + self.incorrect + self.partial + self.missing
        self.actual = self.correct + self.incorrect + self.partial + self.spurious
        self.precision = self.correct / self.actual
        self.recall = self.correct / self.possible
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

    def plot(self):
        res_df = pd.DataFrame(
            {
                "names": ["Precision", "Recall", "F1"],
                "values": np.round(np.array([self.precision, self.recall, self.f1]), 2),
            }
        )
        dodge_text = p9.position_dodge(width=0.9)
        ccolor = "black"
        bar_color = "#130069"

        p = (
            ggplot(res_df, aes(x="names", y="values"))
            + p9.geom_col(
                stat="identity", position="dodge", show_legend=False, fill=bar_color
            )
            + p9.lims(y=(0, 1))
            + p9.geom_text(
                aes(label="values"),
                position=dodge_text,
                size=11,
                va="bottom",
                format_string="{}",
            )
            + p9.theme(
                panel_background=p9.element_rect(fill="white"),
                axis_title_y=p9.element_blank(),
                axis_title_x=p9.element_blank(),
                axis_line_x=p9.element_line(color="black"),
                axis_line_y=p9.element_blank(),
                axis_text_y=p9.element_blank(),
                axis_text_x=p9.element_text(color=ccolor),
                axis_ticks_major_y=p9.element_blank(),
                panel_grid=p9.element_blank(),
                panel_border=p9.element_blank(),
            )
        )

        return p
