# 15. Deepfake tweets detection
## Project I Proposal

This subfolder contains the source code and documentation for project 1 PoC for the NLP course, MiNI PW 2023.

Authors:
- Adam Frej
- Adrian Kamiński
- Piotr Marciniak
- Szymon Szmajdziński

### Workspace Structure
```
├── data/
│   ├── README.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_cleaning_preprocessing_dataset.ipynb
│   ├── 03_stemming_lemmatization.ipynb
│   ├── 04_bert_word_embeddings_ml.ipynb
│   ├── 11_modeling_tfidf.ipynb
│   ├── 21_CharCNN.ipynb
│   ├── 22_CharGRU.ipynb
│   ├── 23_CharCNN+GRU.ipynb
├── utils/
│   ├── KerasModels.py
│   ├── optuna_utils.py
│   └── __init__.py
└── README.md
```


### Directory Descriptions
- [data/](./data/): This directory contains the original datasets used in this project. It includes training, testing, and validation datasets.
- [notebooks/](./notebooks/): This directory contains all the Jupyter notebooks used for exploratory data analysis, data cleaning and preprocessing, stemming and lemmatization, and modeling.
- [utils/](./utils/): This directory contains Python scripts for Keras models and Optuna utilities.
- [README.md](./README.md): This file provides an overview of the project and describes the workspace structure.
