import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import streamlit_scrollable_textbox as stx

import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from collections import Counter
from utils import PATH_TO_DATA, plotly_config
from utils import count_lemmas, prepare_lemmas_counter_dataframe
from plots import prepare_barplot_words


@st.cache_data(show_spinner="Reading the speech...")
def load_data(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


def prepare_lexical_dispersion_plot(lemmas, words_to_show):
    color_map = {
        word: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, word in enumerate(words_to_show)
    }

    # Filter out words not in words_to_show
    lemmas_filtered = [word if word in words_to_show else None for word in lemmas]

    # Create a list of colors and alpha values for each lemma
    colors = [color_map.get(word, "rgba(0,0,0,0)") for word in lemmas_filtered]
    alpha_values = [0.5 if word is not None else 0 for word in lemmas_filtered]

    # Create a scatter plot
    fig = go.Figure(
        go.Scatter(
            x=[pos if word is not None else None for pos, word in enumerate(lemmas_filtered)],
            y=[1] * len(lemmas_filtered),
            mode="markers",
            marker=dict(color=colors, opacity=alpha_values, size=10),
            text=[
                f"Word: <b>{word}</b><br>Position (tokens): <b>{pos}</b><br>Relative position: <b>{pos / len(lemmas):.3f}</b>"
                for pos, word in enumerate(lemmas)
            ],
            hoverinfo="text",
        )
    )

    # Update layout
    fig.update_layout(
        yaxis=dict(showline=False, showticklabels=False, showgrid=False),
        xaxis=dict(title="Word Position"),
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
    )

    return fig


st.title("Speech Viewer")


df = load_data(PATH_TO_DATA)


with st.sidebar:
    year = st.slider("Year", 1946, 2023, (2020), step=1, format="%d")
    countries = st.selectbox("Country", np.sort(df["Country"].unique()))
    number_of_words = st.slider("Number of words", 1, 50, (15), step=1, format="%d")
    words_to_remove = st.text_input("Additional words to remove", "", placeholder="Enter words separated by comma")
    words_to_remove = [word.strip() for word in words_to_remove.split(",")]

    selected_row = df.loc[(df["Year"] == year) & (df["Country"] == countries), :]


if len(selected_row) > 0:
    text = selected_row["text"].values[0]
    lemmas = selected_row["lemmas"].values[0]
    top_lemmas_with_count = count_lemmas(lemmas, number_of_words, words_to_remove)
    top_lemmas_words = [word for word, count in top_lemmas_with_count]

    cols_main1 = st.columns(2)
    cols_main1[0].metric("Speaking Person", selected_row["Name of Person Speaking"].values[0])
    cols_main1[1].metric("Position of Speaking Person", selected_row["Post"].values[0])
    cols_main2 = st.columns(2)
    cols_main2[0].metric("Region Name", selected_row["Region Name"].values[0])
    cols_main2[1].metric("Sub-region Name", selected_row["Sub-region Name"].values[0])
    style_metric_cards(
        background_color="#9de1fd",
        border_left_color="#0499d4",
    )

    with st.expander("Show metadata"):
        cols_row1 = st.columns(4)
        cols_row1[0].metric("Population", f"{selected_row['Population'].values[0]:,.0f}")
        cols_row1[1].metric("Total Fertility Rate", f"{selected_row['TFR'].values[0]:.2f}")
        cols_row1[2].metric("Human Development Index", f"{selected_row['HDI'].values[0]:.2f}")
        cols_row1[3].metric("GDP (constant 2015 US$)", f"{selected_row['GDP'].values[0]/1000:,.0f} K")

        cols_row2 = st.columns(4)
        cols_row2[0].metric("Unemployment Rate", f"{selected_row['Unemployment Rate'].values[0]:.2f}")
        cols_row2[1].metric("Gini Index", f"{selected_row['Gini'].values[0]:.2f}")
        cols_row2[2].metric("CO2 Emissions per Capita [t]", f"{selected_row['CO2'].values[0]:,.2f}")
        cols_row2[3].metric("Democracy Index", f"{selected_row['Democracy Index'].values[0]:.2f}")

    st.subheader("The most common words")
    st.plotly_chart(
        prepare_barplot_words(prepare_lemmas_counter_dataframe(top_lemmas_with_count)),
        use_container_width=True,
        config=plotly_config,
    )

    st.subheader("Lexical dispersion plot")
    words_to_show = st.multiselect("Words to show", top_lemmas_words, default=top_lemmas_words[:5])
    st.plotly_chart(
        prepare_lexical_dispersion_plot(lemmas, words_to_show), use_container_width=True, config=plotly_config
    )

    st.subheader("Text of the speech")
    stx.scrollableTextbox(text, height=400)

else:
    st.warning("No data for this year and country")
