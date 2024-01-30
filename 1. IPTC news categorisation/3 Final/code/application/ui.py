import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import random

st.title("IPTC Categorisation App")

article_text = st.text_area("Enter the text of the article:", "")

selected_hierarchies = st.multiselect("Select Hierarchies", [1, 2, 3], default=[1])

num_results = st.number_input("Number of Results", min_value=1, value=10)

if st.button("Submit"):
    if article_text:
        data = {
            'article_text': article_text,
            'selected_hierarchies': selected_hierarchies,
            'n_results': num_results
        }

        response = requests.post('http://flask-app:6004/perform_query', json=data)

        if response.status_code == 200:
            api_results = response.json()

            st.title("Best matching IPTC categories")
            st.table({'IPTC Name': api_results['names'], 'Hierarchy': api_results['hierarchies'], 'Distance': api_results['distances']})
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
        
        data2 = {
            'article_text': article_text,
            'selected_hierarchies': selected_hierarchies,
            'n_results': 10
        }
        
        response2 = requests.post('http://flask-app:6004/get_most_similar_articles', json=data2)

        if response2.status_code == 200:
            api_results2 = response2.json()
            articles = api_results2['articles'][0]
            articles_short = [article[:100] + "..." for article in articles]
            
            st.title("Best matching articles")
            st.table({'Article': articles_short, 'IPTC name': (x['name'] for x in api_results2['metadatas'][0])})

            iptc_name_counts = {}
            for x in api_results2['metadatas'][0]:
                if x['name'] in iptc_name_counts:
                    iptc_name_counts[x['name']] += 1
                else:
                    iptc_name_counts[x['name']] = 1
            
            st.title("Distribution of best matching articles by IPTC name")
            fig, ax = plt.subplots()
            colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in iptc_name_counts]

            ax.bar(iptc_name_counts.keys(), iptc_name_counts.values(), color=colors)
            
            plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.set_ylabel('Count')
            ax.set_xlabel('IPTC Name')
            st.pyplot(fig)

        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter the text of the article before submitting.")
