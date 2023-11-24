# nlp-project

## Goal

We have a set of news articles in Slovenian and English language. We want to categorize them according to the IPTC taxonomy.

## Solution Overview

Our approach uses pretrained versatile language model embeddings to represent news articles:
1. We generate embeddings for all articles in the dataset, using a pretrained Large Language Model. 

    These sort of embeedings capture the semantic meaning of the whole text fragment and can be used in multilingual setting depending on the model.
    For the sake of fast prototyping, we decided to apply [OpenAI ADA Embeddings](https://platform.openai.com/docs/guides/embeddings), which are still considered state of the art across variety of NLP tasks according to results from 2023 paper [“MTEB: Massive text embedding benchmark”](https://arxiv.org/abs/2210.07316).

2. We generate embedding for each category name.

3. We calculate cosine similarity between each article and each category name. The category with the highest similarity is assigned to the article.


The solution so far achieves around 86% accuracy on the test set in the PoC phase.


Further work:
- We plan to try generating embeddings for category descriptions instead of category names. This has a chance to increase the precision of the categorization.
- We plan to use language models to generate a set of diversified descriptions for each category, based on the original description. Then we can generate their embeddings and approach the whole problem through as a sort of voting process. The are many ways to approach this so the details of such solutions are yet to be defined.


## Project structure

Project folders:
1. EDA - contains exploratory data analysis of datasets and categorization results.
2. Classes - contains classes and modules used in the project.
3. Data - contains datasets used in the project.
4. Labeling_app - contains an app to label news articles, developed by our team.
5. Tests - contains POC level code for modeling and classification. 


```bash
|-- EDA
|   |-- exploratory_data_analysis.ipynb
|   `-- explore_IPTC_categorisation_results.ipynb
|-- classes
|   |-- __init__.py
|   |-- article_data_handler.py
|   |-- embedding_visualizer.py
|   |-- embeddings.py
|   `-- exploratory_data_analysis.py
|-- data
|   |-- 2023_articles_en
|   |   |-- slovenian articles ...
|   |-- 2023_articles_sk
|   |   |-- english articles ...
|   |-- API_conn.ipynb
|   |-- articles_2023_en.csv
|   |-- cosine_similarity.csv
|   |-- labeled_data
|   |-- pickle
|   |   |-- data_2023_10_29.pickle
|   |   |-- list_of_ids_2023.pickle
|   |   `-- list_of_ids_en_2023.pickle
|   |-- taxonomy
|   |   |-- IPTC-labels_table.csv
|   |   |-- TAKSONOMIJA.xlsx
|   |   `-- taxonomy.csv
|   `-- test_sets
|   |   |-- test_set_balanced.csv   
|   |   `-- test_set_stratified.csv
`-- tests
|   |-- test_embedding_visualizations.ipynb
|   |-- test_openai_embedding.ipynb
|   `-- test_taxonomy.ipynb
|-- README.md
```
