import streamlit as st
import streamlit as st

import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards

import streamlit_scrollable_textbox as stx

from utils import PATH_TO_DATA, plotly_config, text_descriptive_attributes_mapping, metadata_vars_mapping

variables_dict = {**metadata_vars_mapping, **text_descriptive_attributes_mapping}


def prepare_attribute_values(tmp_df):
    for variable_name, variable_df_name in variables_dict.items():
        st.metric(variable_name, f"{np.round(tmp_df[variable_df_name].values[0],2)}")


@st.cache_data(show_spinner="Reading the speeches...")
def load_data(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


st.title("Speech Comparer")
df = load_data(PATH_TO_DATA)


cols = st.columns(2)
with cols[0]:
    st.subheader("First Speech")
    speech1_year = st.slider("Year", 1946, 2023, (2020), step=1, format="%d")
    speech1_country = st.selectbox("Country", np.sort(df["Country"].unique()))
    tmp1 = df[(df["Year"] == speech1_year) & (df["Country"] == speech1_country)]
    if len(tmp1) == 0:
        st.error("No speeches found for the selected year and country.")
    else:
        st.markdown(
            f"""
            #### Speaking Person
            **{tmp1.iloc[0]["Name of Person Speaking"]}**,
            *{tmp1.iloc[0]["Post"]}*\n
                """,
            unsafe_allow_html=True,
        )

        st.markdown("### Speech Text")
        stx.scrollableTextbox(tmp1.iloc[0]["text"], height=500, key="speech1_text")

        st.markdown("### Attributes")
        prepare_attribute_values(tmp1)


with cols[1]:
    st.subheader("Second Speech")
    speech2_year = st.slider("Year", 1946, 2023, (2020), step=1, format="%d", key="speech2_year")
    speech2_country = st.selectbox("Country", np.sort(df["Country"].unique()), index=2, key="speech2_country")
    tmp2 = df[(df["Year"] == speech2_year) & (df["Country"] == speech2_country)]
    if len(tmp2) == 0:
        st.error("No speeches found for the selected year and country.")
    else:
        st.markdown(
            f"""
            #### Speaking Person
            **{tmp2.iloc[0]["Name of Person Speaking"]}**,
            *{tmp2.iloc[0]["Post"]}*\n
                """,
            unsafe_allow_html=True,
        )

        st.markdown("### Speech Text")
        stx.scrollableTextbox(tmp2.iloc[0]["text"], height=500, key="speech2_text")

        st.markdown("### Attributes")
        prepare_attribute_values(tmp2)
