import streamlit as st
from st_pages import show_pages_from_config

st.set_page_config(layout="wide")
show_pages_from_config()

st.title("UNGA Debates Explorer")

st.header("About this app")

st.markdown(
    """
Welcome to the **UNGA Debates Explorer**, an interactive application developed to enable easy exploration of United Nations General Assembly Debates corpus. This user-friendly tool, built on the Streamlit data app framework, provides a dynamic platform for analyzing the speeches, complete with metadata and statistics.

The application is designed to allow you to navigate through this corpus efficiently, and conduct analyses tailored to specific dimensions of interest. The app is structured to allow you to customize your analyses, selecting specific timeframes, countries, or variables. Results are predominantly presented through visualizations, enhancing the interpretability of results and facilitating the communication of findings.
"""
)

st.header("Key Features")

st.markdown(
    """
1. **Speech Viewer**: This page allows you to view the full text of a speech selected by year and country. 
It displays the speech text alongside relevant metadata and basic analyses, including the most common words and a lexical dispersion plot. 

2. **Speech Comparer**: This page allows for a side-by-side comparison of two speeches, their metadata, and calculated  statistics.

3. **Analysis Over Years**: This page features visualizations that illustrate analyses over the selected time period 
for chosen groups of countries. You can explore trends in the most common words and observe how measures change over time. 
Such analyses reflect the evolving themes and priorities in the UNGA debates over different epochs.

4. **Speech Attributes**: This page allows for the in-depth exploration of the collected metadata and calculated descriptive statistics. 
You can compare values of various selected variables across different years and countries, 
thereby gaining a multifaceted perspective on the interplay between linguistic elements and contextual attributes. 
Moreover, this tab facilitates an examination of how the frequency of a chosen word correlates with metadata variables describing the countries.

5. **BERTopic Analysis**: This page allows for the exploration of the topic modeling results. They can be explored using the following visualizations:
    - Intertopic distance map, which allows to compare the relative sizes and distances between the topics.
    - Similarity matrix, which allows to compare the most similar topics based on their words.
    - Dendrogram, which allows to view the hierarchical topic structure.
    - Line plot showing the dynamics and evolution of topics over time.
"""
)

st.header("About the Data")
st.markdown(
    """
Our dataset is derived from the [United Nations General Debate Corpus 1946-2022](https://doi.org/10.7910/DVN/0TJX8Y), covering speeches from 1946 to 2022. With over 10,000 speeches from representatives of 193+ countries, it offers valuable metadata, including the country, speaker's name, and position.

In our update, we integrated speeches from 2023, rectified errors in metadata, and enriched the dataset with additional variables and indices, providing a more comprehensive understanding of countries and their situations across different years.

Additionaly, for each speech, we generated a list of the most frequent words with their counts, 
accompanied by various descriptive statistics. These include metrics like the number of tokens, unique tokens, sentences, and characters,
measures such as the mean, median, and standard deviation of token and sentence length, as well as
various readability metrics, part-of-speech proportions, and document coherence values.
"""
)

st.header("Authors")
st.markdown(
    """
This application was developed by: 

- M. Grzyb
- M. Krzyziński
- M. Spytek 
- B. Sobieski

It is a part of the project for the Natural Language Processing course at the Warsaw University of Technology.
"""
)
