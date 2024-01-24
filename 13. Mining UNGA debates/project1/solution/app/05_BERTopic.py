import streamlit as st
from bertopic import BERTopic
import os

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards

import streamlit_scrollable_textbox as stx

from utils import (
    PATH_TO_MODEL,
    PATH_TO_CORPUS,
    get_texts,
    plotly_config,
    text_descriptive_attributes_mapping,
    metadata_vars_mapping,
)

N_BINS = 76

variables_dict = {**metadata_vars_mapping, **text_descriptive_attributes_mapping}


def prepare_attribute_values(tmp_df):
    for variable_name, variable_df_name in variables_dict.items():
        st.metric(variable_name, f"{np.round(tmp_df[variable_df_name].values[0],2)}")


@st.cache_data(show_spinner="Loading the model...")
def load_model_and_texts(path):
    topic_model = BERTopic.load(path, embedding_model="all-mpnet-base-v2")
    all_texts = get_texts(PATH_TO_CORPUS)
    return topic_model, all_texts


st.title("Topic modeling with BERTopic")


topic_model, all_texts = load_model_and_texts(PATH_TO_MODEL)


# read in topics_over_time from pickle if it exists:
if os.path.exists("../topics_over_time.pickle"):
    topics_over_time = pd.read_pickle("../topics_over_time.pickle")
else:
    texts = [text[0] for text in all_texts]
    timesteps = [int(text[1]["year"]) for text in all_texts]
    topics_over_time = topic_model.topics_over_time(texts, timesteps, nr_bins=N_BINS)
    topics_over_time.to_pickle("../topics_over_time.pickle")


with st.sidebar:
    selected_topics = st.multiselect(
        "Select topics",
        [
            str(x) + " " + str(y)
            for x, y in zip(
                topic_model.get_topic_info().Topic,
                topic_model.get_topic_info().Representation,
            )
        ],
    )

selected_topic_ids = [int(x.split(" ")[0]) for x in selected_topics] if selected_topics else [i for i in range(0, 10)]

st.plotly_chart(topic_model.visualize_topics(width=800), config=plotly_config)
st.plotly_chart(
    topic_model.visualize_topics_over_time(topics_over_time, topics=selected_topic_ids),
    config=plotly_config,
    use_container_width=True,
)

st.plotly_chart(topic_model.visualize_heatmap(topics=selected_topic_ids, width=800), config=plotly_config)
st.plotly_chart(topic_model.visualize_hierarchy(topics=selected_topic_ids, width=800), config=plotly_config)

st.plotly_chart(topic_model.visualize_barchart(topics=selected_topic_ids, n_words=7), config=plotly_config)
