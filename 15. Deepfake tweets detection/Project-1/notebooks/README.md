# Notebooks

This directory contains various Jupyter notebooks and Python scripts used in the project.

## Jupyter Notebooks

- [`01_EDA.ipynb`](01_EDA.ipynb): Contains exploratory data analysis of the TweepFake dataset.
- [`01_EDA-gpt-2-output.ipynb`](01_EDA-gpt-2-output.ipynb): Contains exploratory data analysis of the GPT-2 output dataset.
- [`02_cleaning_preprocessing_dataset.ipynb`](02_cleaning_preprocessing_dataset.ipynb): Contains the data cleaning and preprocessing steps.
- [`03_stemming_lemmatization.ipynb`](03_stemming_lemmatization.ipynb): Contains the stemming and lemmatization process.
- [`04_bert_word_embeddings_ml.ipynb`](04_bert_word_embeddings_ml.ipynb): Contains the process of using BERT for word embeddings and creation BERT dataset.
- [`11_modeling_tfidf.ipynb`](11_modeling_tfidf.ipynb): Contains the process of modeling using TF-IDF which heavily uses Optuna utils.
- [`12_modeling_bert.ipynb`](12_modeling_bert.ipynb): Contains the process of modeling using BERT embeddings which heavily uses Optuna utils.
- [`21_CharCNN.ipynb`](21_CharCNN.ipynb): Contains the CharCNN model (training and evaluation).
- [`22_CharGRU.ipynb`](22_CharGRU.ipynb): Contains the CharGRU model (training and evaluation).
- [`23_CharCNN+GRU.ipynb`](23_CharCNN%2BGRU.ipynb): Contains the CharCNN+GRU model (training and evaluation).
- [`31_WordCNN.ipynb`](31_WordCNN.ipynb): Contains the WordCNN model (training and evaluation).
- [`32_WordGRU.ipynb`](32_WordGRU.ipynb): Contains the WordGRU model (training and evaluation).
- [`33_WordCNN+GRU.ipynb`](33_WordCNN%2BGRU.ipynb): Contains the combined WordCNN+GRU model (training and evaluation).
- [`41_1-RoBert.ipynb`](41_1-RoBert.ipynb): Contains the training of xlm-roberta-base model with frozen roberta part.
- [`41_2-RoBert.ipynb`](41_2-RoBert.ipynb): Contains the training of xlm-roberta-base model (all elements are unfrozen).
- [`41_3-RoBert.ipynb`](41_3-RoBert.ipynb): Contains the training of xlm-roberta-base model with lower learning rate than in notebook above.
- [`42_1-DistilBert.ipynb`](42_1-DistilBert.ipynb): Contains the training of distilbert-base-uncased model.
- [`42_2-DistilBert+gpt-output.ipynb`](42_2-DistilBert%2Bgpt-output.ipynb): Contains the training of distilbert-base-uncased model with additional initialization step of finetuning on GPT-2 output dataset
- [`43_1-GPT2.ipynb`](43_1-GPT2.ipynb): Contains the training of GPT-2 model.
- [`49_Transformers-Results.ipynb`](49_Transformers-Results.ipynb): Contains the evaluation of transformer models.
- [`90_indepth_comparison.ipynb`](90_indepth_comparing.ipynb): Contains the in-depth comparison of the models. 
It calculates accuracy for different types of generators for ML and DL models.
- [`91_Aggregated_results.ipynb`](91_Aggregated_results.ipynb`): Contains the aggregated results of the models (visualizations and tables).

## Python Scripts

- [`KerasModels.py`](utils%2FKerasModels.py): Contains various Keras models used in the project.
- [`optuna_utils.py`](utils%2Foptuna_utils.py): Contains utilities for using Optuna for hyperparameter optimization. It includes functions to instantiate different classifiers (LGBM, XGB, RandomForest, SVC, Logistic Regression) with parameters suggested by Optuna, and functions to extract the best model and calculate the score.
- [`word_utils.py`](utils%2Fword_utils.py): Contains functions to prepare data, prepare text vectorizer, and get different Keras models (WordCNN, WordGRU, WordCNN+GRU).