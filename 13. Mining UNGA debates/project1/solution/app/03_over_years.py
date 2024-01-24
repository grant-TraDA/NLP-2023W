import streamlit as st

import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


from utils import PATH_TO_DATA, plotly_config, text_descriptive_attributes_mapping
from utils import count_lemmas, prepare_lemmas_counter_dataframe
from plots import prepare_barplot_words


@st.cache_data(show_spinner="Reading the speeches...")
def load_data(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


st.title("Corpora Analysis")
df = load_data(PATH_TO_DATA)


def prepare_plot_over_years(df, y_axis):
    fig = go.Figure()

    for year in df["Year"].unique():
        data_for_year = df[df["Year"] == year]
        fig.add_trace(
            go.Box(
                x=[year] * len(data_for_year),
                y=data_for_year[text_descriptive_attributes_mapping[y_axis]],
                name=str(year),
                marker=dict(color="#0499D4", size=6, opacity=0.5),
                hovertemplate="%{y}<br>Country: %{text}",
                text=data_for_year["Country"],
                boxpoints="all",
            )
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=y_axis,
        font=dict(size=18, color="#7f7f7f"),
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
        showlegend=False,
    )

    return fig


with st.sidebar:
    year_range = st.slider("Year", 1946, 2023, (2000, 2020), step=1, format="%d")
    subregions = st.multiselect(
        "Subregion",
        np.sort(df["Sub-region Name"].astype(str).unique()),
        np.sort(df["Sub-region Name"].astype(str).unique())[:-1],
    )

    subset_speeches = df.loc[
        (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1]) & (df["Sub-region Name"].isin(subregions))
    ]


if len(subset_speeches) > 0:
    all_lemmas = [lemma for lemmas_list in subset_speeches["lemmas"] for lemma in lemmas_list]

    st.subheader("The most common words")
    number_of_words = st.slider("Number of words", 1, 50, (15), step=1, format="%d")
    words_to_remove = st.text_input("Additional words to remove", "", placeholder="Enter words separated by comma")
    words_to_remove = [word.strip() for word in words_to_remove.split(",")]
    top_lemmas_with_count = count_lemmas(all_lemmas, number_of_words, words_to_remove)
    top_lemmas_words = [word for word, count in top_lemmas_with_count]

    st.plotly_chart(
        prepare_barplot_words(prepare_lemmas_counter_dataframe(top_lemmas_with_count)),
        use_container_width=True,
        config=plotly_config,
    )

    st.subheader("Statistics of speeches over years")
    var_to_show = st.selectbox("Value to show", list(text_descriptive_attributes_mapping.keys()))

    st.plotly_chart(
        prepare_plot_over_years(subset_speeches, var_to_show),
        use_container_width=True,
        config=plotly_config,
    )


else:
    st.warning("No data for selected filters")
