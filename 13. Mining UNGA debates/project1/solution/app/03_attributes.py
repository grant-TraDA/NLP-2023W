import streamlit as st

import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


from utils import PATH_TO_DATA, plotly_config, text_descriptive_attributes_mapping, metadata_vars_mapping

variables = list(text_descriptive_attributes_mapping.keys()) + list(metadata_vars_mapping.keys())
variables_dict = {**text_descriptive_attributes_mapping, **metadata_vars_mapping}


@st.cache_data(show_spinner="Reading the speeches...")
def load_data(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


st.title("Speech Attributes")
df = load_data(PATH_TO_DATA)


def prepare_pointplot(sub_df, var1, var2):
    fig = px.scatter(
        sub_df,
        x=variables_dict[var1],
        y=variables_dict[var2],
        color="Sub-region Name",
        hover_name="Country",
        hover_data=["Name of Person Speaking", "Post"],
        color_discrete_sequence=px.colors.qualitative.Light24,
    )

    hover_template = (
        "Country: <b>%{hovertext}</b><br>"
        "Speaker: <b>%{customdata[0]}</b><br>"
        "Post: <b>%{customdata[1]}</b><br>"
        "Year: <b>%{customdata[2]}</b><br>"
        f"{var1}: <b>%{{x:.2f}}</b><br>"
        f"{var2}: <b>%{{y:.2f}}</b><br>"
    )

    fig.update_traces(
        marker_opacity=0.5,
        hovertemplate=hover_template,
        customdata=sub_df[["Name of Person Speaking", "Post", "Year"]].values.tolist(),
    )

    fig.update_layout(
        xaxis_title=var1,
        yaxis_title=var2,
        font=dict(size=18, color="#7f7f7f"),
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
    )

    return fig


def prepare_word_pointplot(sub_df, word, var2):
    sub_df["count"] = sub_df["lemmas"].apply(lambda x: x.count(word))
    fig = px.scatter(
        sub_df,
        x=variables_dict[var2],
        y="count",
        color="Sub-region Name",
        hover_name="Country",
        hover_data=["Name of Person Speaking", "Post"],
        color_discrete_sequence=px.colors.qualitative.Light24,
    )

    hover_template = (
        "Country: <b>%{hovertext}</b><br>"
        "Speaker: <b>%{customdata[0]}</b><br>"
        "Post: <b>%{customdata[1]}</b><br>"
        "Year: <b>%{customdata[2]}</b><br>"
        f"{var2}: <b>%{{y:.2f}}</b><br>"
        f"Count of {word}: <b>%{{customdata[3]}}</b><br>"
    )

    fig.update_traces(
        marker_opacity=0.5,
        hovertemplate=hover_template,
        customdata=sub_df[["Name of Person Speaking", "Post", "Year", "count"]].values.tolist(),
    )

    fig.update_layout(
        xaxis_title=var2,
        yaxis_title=f"Count of {word}",
        font=dict(size=18, color="#7f7f7f"),
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
    )

    return fig


with st.sidebar:
    year_range = st.slider("Year", 1946, 2023, (2000, 2020), step=1, format="%d")
    subset_speeches = df.loc[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]


if len(subset_speeches) > 0:
    cols = st.columns(2)
    with cols[0]:
        var1 = st.selectbox("Feature X", variables, index=0)
    with cols[1]:
        var2 = st.selectbox("Feature Y", variables, index=1)

    st.subheader(f"{var1} vs {var2}")
    st.plotly_chart(
        prepare_pointplot(subset_speeches, var1, var2),
        use_container_width=True,
        config=plotly_config,
    )

    cols2 = st.columns(2)
    with cols2[0]:
        var12 = st.text_input("Word to count in speeches", "peace")
    with cols2[1]:
        var22 = st.selectbox("Feature Y", variables, index=1, key="y2")

    st.subheader(f"Count of {var12} vs {var22}")
    st.plotly_chart(
        prepare_word_pointplot(subset_speeches, var12, var22),
        use_container_width=True,
        config=plotly_config,
    )

else:
    st.warning("No data for selected filters")
