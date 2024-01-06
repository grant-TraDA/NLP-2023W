# Early detection of fake news

The Comparison of Local and Global Early Fake News Detection Methods

## Goals
1. Comparison of different topic detection models.
2. Comparison of fake news detection methods.
3. Introduction of local fake news detection methods.
4. Evaluation of the local approach, and comparison to corresponding global solutions.
5. Exploration of models differences between the two strategies with the usage of XAI.

## Dataset

In the project we used a dataset called **Fake News Corspus**, which can be downloaded from the following site: [https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0](https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0)

## Solution overview

![](./img/proposed_solution.jpg)

## Repository structure

```
├── img                                     # plot with solution overview
│   └── proposed_solution.jpg
├── MS1                                     # files for MS1 - report and presention of project proposition
│   ├── Early detection of fake news based on discussion topic.pdf
│   └── NLP_Project_Report___Early_Fake_Detection.pdf
├── MS2                                     # files for MS2 - Proof of Concept
│   ├── clustering_models.ipynb             # 03 - initial clustering
│   ├── data_preprocessing.py               # scripts for data preprocessing
│   ├── EDA_final.ipynb                     # 02 - final EDA
│   ├── eda_for_nlp_package.py              # scripts for EDA
│   ├── initial_EDA.ipynb                   # 01 - initial EDA
│   └── PoC_modelling.ipynb                 # 04 - initial models
├── MS3                                     # files for MS3 - final solution
│   ├── 01-EDA                              # 01 - EDA - data, scripts and preprocessed files
│   │   ├── data_1000 ...
│   │   ├── data_preprocessing_development.py
│   │   ├── data_preprocessing.py
│   │   ├── EDA_final.ipynb
│   │   ├── eda_for_nlp_package.py
│   │   └── initial_EDA.ipynb
│   ├── 02-clustering                       # 02 - clustering - scripts and preprocessed files
│   │   ├── clustering_models.ipynb
│   │   ├── main_df.csv -> ../01-EDA/data_1000/main_df.csv
│   │   └── outputs ...
│   ├── 03-models                           # 03 - models - script and files with trained models
│   │   ├── LaTeX ...
│   │   ├── modelling.ipynb                 # final models
│   │   ├── models ...
│   │   ├── PoC modelling.ipynb             # POC models
│   │   └── preprocessed_datasets
│   │       ├── gssdmm7_nouns.csv -> ../../02-clustering/outputs/gssdmm7_nouns.csv
│   │       ├── ldatfidf4_nouns.csv -> ../../02-clustering/outputs/ldatfidf4_nouns.csv
│   │       ├── main_df.csv -> ../../01-EDA/data_1000/main_df.csv
│   │       └── README.md
│   ├── 04-explainability                   # 04 - explainability - scripts for explanations with files
│   │   ├── explainability.ipynb
│   │   ├── explanations ...
│   │   ├── models -> ../03-models/models
│   │   └── preprocessed_datasets
│   │       ├── gssdmm7_nouns.csv -> ../../02-clustering/outputs/gssdmm7_nouns.csv
│   │       ├── ldatfidf4_nouns.csv -> ../../02-clustering/outputs/ldatfidf4_nouns.csv
│   │       └── main_df.csv -> ../../01-EDA/data_1000/main_df.csv
│   ├── presentation.pdf                    # presentation
│   └── report.pdf                          # report
└── README.md
```
