# IPTC categorisation application

This is a contenerized application for categorisation of press news articles. To run the application, type the following command in terminal:

```bash

 docker-compose up --build

```

Then, you can either use Streamlit User Interface on `localhost:8501`, or use the API endpoints.

# ChromaDB Flask Server API

## 1. Perform Query Endpoint

**Endpoint:** `POST /perform_query`

This endpoint allows you to perform a query on the ChromaDB server, in order to find most fitting IPTC categories.

### Request

- **Method:** `POST`
- **URL:** `http://localhost:6004/perform_query`
- **Headers:**
  - Content-Type: application/json

**Request Body:**
```json
{
  "article_text": "Sample article text",
  "selected_hierarchies": [1, 2, 3],
  "n_results": 5
}
```
## 2. Get Most Similar Articles Endpoint

**Endpoint:** `POST /get_most_similar_articles`

This endpoint allows you to get most similar articles already present in ChromaDB.

### Request

- **Method:** `POST`
- **URL:** `http://localhost:6004/get_most_similar_articles
- **Headers:**
  - Content-Type: application/json

**Request Body:**
```json
{
  "article_text": "Sample article text",
  "selected_hierarchies": [1, 2, 3],
  "n_results": 5
}
```

## 3. Insert To Database Endpoint

**Endpoint:** `POST /insert_to_database`

This endpoint allows you to insert article into the ChromaDB.

### Request

- **Method:** `POST`
- **URL:** `http://localhost:6004/insert_to_database`
- **Headers:**
  - Content-Type: application/json

**Request Body:**
```json
{
  "id": "123",
  "document": "This is a sample document.",
  "metadata": {"key": "value"}
}
```
