import streamlit as st
import chromadb

# Connect to the ChromaDB. First, set up the ChromaDB server with command `chroma run --path ./chroma`.
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection("vector_db")

# Function to perform a query and return results
def perform_query(query_text, hierarchies, n_results=5):
    where_clause = {"hierarchy": {"$in": hierarchies}}
    query_result = collection.query(query_texts=[query_text], n_results=n_results, where=where_clause)

    names = []
    distances = []
    hierarchies = []

    for i in range(len(query_result['ids'][0])):
        names.append(query_result['metadatas'][0][i]['name'])
        distances.append(query_result['distances'][0][i])
        hierarchies.append(query_result['metadatas'][0][i]['hierarchy'])

    return names, distances, hierarchies

# Streamlit app
st.title("ChromaDB Query App")

# User input for article text
article_text = st.text_area("Enter the text of the article:", "")

# Checkbox for hierarchies
selected_hierarchies = st.multiselect("Select Hierarchies", [1, 2, 3], default=[1])

# Input field for the number of results
num_results = st.number_input("Number of Results", min_value=1, value=5)

# Button to submit the text and perform the query
if st.button("Submit"):
    if article_text:
        # Perform the query and get results
        names, distances, hierarchies = perform_query(article_text, selected_hierarchies, n_results=num_results)

        # Display results in a table
        st.table({'IPTC Name': names, 'Hierarchy': hierarchies, 'Distance': distances})
    else:
        st.warning("Please enter the text of the article before submitting.")
