from flask import Flask, request, jsonify
import chromadb
import time
import pandas as pd
import uuid 
import os
from chromadb.utils import embedding_functions

app = Flask(__name__)

def wait_for_chroma():
    """
    Wait for Chroma to be available before running the app.
    """
    max_retries = 30 
    retries = 0

    while retries < max_retries:
        try:
            # Try to connect to Chroma
            chromadb.HttpClient(host='chroma', port=8000)
            print("Connected to Chroma successfully!")
            return
        except Exception as e:
            print(f"Error connecting to Chroma: {e}")
            retries += 1
            time.sleep(2)

    raise ConnectionError("Unable to connect to Chroma after multiple attempts.")


wait_for_chroma()

client = chromadb.HttpClient(host='chroma', port=8000)

hugging_face_key = os.environ['HUGGING_FACE_KEY']
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=hugging_face_key,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create the collection if it doesn't exist
try:
    collection = client.get_collection("iptc-categories", embedding_function=huggingface_ef)

except:

    taxonomy_chroma = pd.read_csv('./taxonomy_chroma.csv')

    hierarchy = list(taxonomy_chroma['hierarchy'])
    name = list(taxonomy_chroma['name'])
    name_code = list(taxonomy_chroma.index)
    documents = list(taxonomy_chroma['final_description'])
    ids = [str(uuid.uuid4()) for _ in range(len(name))]

    metadata = {
        'name': name,
        'name_code': name_code,
        'hierarchy': hierarchy
    }

    metadatas = [{k: v[i] for k, v in metadata.items()} for i in range(len(metadata['name']))]

    collection = client.create_collection("iptc-categories", embedding_function=huggingface_ef)
    collection.add(
        ids = ids,
        documents = documents,
        metadatas = metadatas
    )

try:
    collection2 = client.get_collection("articles", embedding_function=huggingface_ef)

except:

    collection2 = client.create_collection("articles", embedding_function=huggingface_ef)
    articles_chroma = pd.read_csv('./articles_chroma.csv')

    embeddings = list(articles_chroma['embeddings'].apply(lambda x: eval(x)))
    texts = list(articles_chroma['text'])
    query_result = collection.query(query_embeddings=embeddings, n_results=1)

    articles = []
    metadatas = []
    embeddings2 = []

    for i in range(len(query_result['ids'])):
        if 0.5 < query_result['distances'][i][0] < 0.9:
            articles.append(texts[i])
            metadatas.append(query_result['metadatas'][i][0])
            embeddings2.append(embeddings[i])

    ids = [str(uuid.uuid4()) for i in range(len(articles))]
    collection2.add(
        ids = ids,
        documents = articles,
        metadatas = metadatas,
        embeddings = embeddings2
    )

    
# API endpoints

@app.route('/perform_query', methods=['POST'])
def get_most_similar_categories():
    request_data = request.json
    
    query_text = request_data.get('article_text', '')
    hierarchies = request_data.get('selected_hierarchies', [])
    n_results = request_data.get('n_results', 5)

    where_clause = {"hierarchy": {"$in": hierarchies}}
    query_result = collection.query(query_texts=[query_text], n_results=n_results, where=where_clause)

    names = []
    distances = []
    hierarchies = []

    for i in range(len(query_result['ids'][0])):
        names.append(query_result['metadatas'][0][i]['name'])
        distances.append(query_result['distances'][0][i])
        hierarchies.append(query_result['metadatas'][0][i]['hierarchy'])

    response_data = {
        'names': names,
        'distances': distances,
        'hierarchies': hierarchies
    }

    return jsonify(response_data)

@app.route('/insert_to_database', methods=['POST'])
def insert_to_database():
    request_data = request.json
    
    id = request_data.get('id', '')
    document = request_data.get('document', '')
    metadata = request_data.get('metadata', '')

    collection.add(
        ids=[id],
        documents=[document],
        metadatas=[metadata]
    )

    return jsonify({})
                   
@app.route('/get_most_similar_articles', methods=['POST'])
def get_most_similar_articles():
    request_data = request.json
    
    query_text = request_data.get('article_text', '')
    hierarchies = request_data.get('selected_hierarchies', [])
    n_results = request_data.get('n_results', 5)

    where_clause = {"hierarchy": {"$in": hierarchies}}
    query_result = collection2.query(query_texts=[query_text], n_results=n_results, where=where_clause)

    response_data = {
        'articles': query_result['documents'],
        'metadatas': query_result['metadatas'],
        'embeddings': query_result['embeddings']
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6004)
