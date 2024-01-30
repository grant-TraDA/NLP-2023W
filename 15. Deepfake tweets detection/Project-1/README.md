# 15. Deepfake tweets detection
## Project I

This subfolder contains the source code for project 1 for the NLP course, MiNI PW 2023.

Authors:
- Adam Frej
- Adrian Kamiński
- Piotr Marciniak
- Szymon Szmajdziński

### Workspace Structure
```
├── data
│   ├── gpt-2-ouput-dataset
│   │   ├── download_dataset.py
│   │   ├── README.md
│   ├── README.md
├── notebooks
│   ├── 01_EDA-gpt-2-output.ipynb
│   ├── 01_EDA.ipynb
│   ├── 02_cleaning_preprocessing_dataset.ipynb
│   ├── 03_stemming_lemmatization.ipynb
│   ├── 04_bert_word_embeddings_ml.ipynb
│   ├── 11_modeling_tfidf.ipynb
│   ├── 12_modeling_bert.ipynb
│   ├── 21_CharCNN.ipynb
│   ├── 22_CharGRU.ipynb
│   ├── 23_CharCNN+GRU.ipynb
│   ├── 31_WordCNN.ipynb
│   ├── 32_WordGRU.ipynb
│   ├── 33_WordCNN+GRU.ipynb
│   ├── 41_1-RoBert.ipynb
│   ├── 41_2-RoBert.ipynb
│   ├── 41_3-RoBert.ipynb
│   ├── 42_1-DistilBert.ipynb
│   ├── 42_2-DistilBert+gpt-output.ipynb
│   ├── 43_1-GPT2.ipynb
│   ├── 49_Transformers-Results.ipynb
│   ├── 90_indepth_comparing.ipynb
│   ├── 91_Aggregate_results.ipynb
│   ├── README.md
│   ├── results
│   │   ├── bert.csv
│   │   ├── bert_optuna.csv
│   │   ├── char_cnn.csv
│   │   ├── char_cnn_gru.csv
│   │   ├── char_gru.csv
│   │   ├── results_in_depth.csv
│   │   ├── results_in_depth-transformers.csv
│   │   ├── tfidf.csv
│   │   ├── tfidf_optuna.csv
│   │   ├── transformers.csv
│   │   ├── word_cnn.csv
│   │   ├── word_cnn_gru.csv
│   │   └── word_gru.csv
│   └── utils
│       ├── __init__.py
│       ├── KerasModels.py
│       ├── optuna_utils.py
│       └── word_utils.py
├── README.md
├── docker-compose.yml
└── requirements.txt
```


### Directory Descriptions
- [data/](./data/): This directory contains the original datasets used in this project. It includes training, testing, and validation datasets.
- [notebooks/](./notebooks/): This directory contains all the Jupyter notebooks used for exploratory data analysis, data cleaning and preprocessing, stemming and lemmatization, and modeling.
- [utils/](./utils/): This directory contains Python scripts for Keras models and Optuna utilities.
- [README.md](./README.md): This file provides an overview of the project and describes the workspace structure.
- [docker-compose.yml](./docker-compose.yml): This file can be used to setup jupyter server environment to run notebooks using GPU.
- [requirements.txt](./requirements.txt): This file is used to store all Python dependencies for the project, which can be then easily installed.

### Results

Results are available in [notebooks/results](./notebooks/results) folder. Model checkpoints and preprocessed datasets can be found in [Google Drive](https://drive.google.com/drive/folders/17fvFoImpwdA98alU3-hE-7qwlPgyBTQn?usp=sharing).
