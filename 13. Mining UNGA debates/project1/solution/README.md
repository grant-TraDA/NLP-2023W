# Team 13. - Mining United Nations General Assembly debates - solution

Mateusz Grzyb, Mateusz Krzyziński, Bartłomiej Sobieski, Mikołaj Spytek

## Where is our solution available?

- The application created during this project is hosted at [https://nlp-unga-debates-g2o6gnzttq-lm.a.run.app/](https://nlp-unga-debates-g2o6gnzttq-lm.a.run.app/), at least until we run out of free credits at the *Google Cloud Platform*.
- The application is also available as a *Docker* image at [https://hub.docker.com/r/krzyzinskim/nlp-unga-debates](https://hub.docker.com/r/krzyzinskim/nlp-unga-debates). To run it, you simply need to run the docker container with appropriate port forwarding.
- The binary pickle file with the processed dataset (used in the app) is available at [https://drive.google.com/drive/folders/16a-Woz3FoPsXd77MsIyIMZkKlwSEIkox?usp=share_link](https://drive.google.com/drive/folders/16a-Woz3FoPsXd77MsIyIMZkKlwSEIkox?usp=share_link). To run the application, it should be placed in the root solution directory.

## Folder structure

The folder structure of the solution is as follows:

```
.
├── analysis # notebooks containing analyses performed using the BERTopic model
├── app # source code of the interactive application for exploring the final results
├── dataset # cleaned version of the previously collected UNGA statements enriched with the 2023 session and additional metadata
├── metadata # raw metadata files and scripts necessary for their preprocessing and joining to the original dataset
├── metrics # source code and results of evaluating the models using topic modeling metrics
├── modeling # source code used for creating the BERTopic models
├── models # model weights for all models considered in the project
├── preprocessing # scripts used for preprocessing the texts
├── scrapping # scripts used for scrapping the 2023 session
├── README.md # Main readme file
└── requirements.txt # Main environment definition
```

## Reproducing the results

### Environment

Start by creating the main Python 3.11 environment e.g. using `conda`:

```
conda create -yn mining_unga_debates python=3.11 &&
conda run -n mining_unga_debates --no-capture-output pip install -r requirements.txt
```

### Dataset

We recommend extracting the prepared dataset and proceeding directly to the modeling step:

```
mkdir corpora &&
unzip -q dataset/dataset.zip -d corpora
```

You also need to download the `data_processed_add_features.pickle` file from [https://drive.google.com/drive/folders/16a-Woz3FoPsXd77MsIyIMZkKlwSEIkox?usp=share_link](https://drive.google.com/drive/folders/16a-Woz3FoPsXd77MsIyIMZkKlwSEIkox?usp=share_link) and put it in the root solution directory.

### Scrapping

To reproduce the statement scrapping step first make sure that you have the browser driver for Selenium installed. For Debian and its derivatives you can install the `chromium-driver` package (`sudo snap install chromium-driver`). For Arch and its derivatives you can install the `chromedriver` package (`sudo pacman -S chromedriver`).

First, scrap the 2023 session's statements in PDF format using:

```
conda run -n mining_unga_debates --no-capture-output python scrapping/download.py
```

It will populate the `corpora/pdfs` directory with PDF files.

Second, extract texts from these statements in TXT format using:

```
conda run -n mining_unga_debates --no-capture-output python scrapping/extract.py
```

It will populate the `corpora/UN General Debate Corpus/TXT/Session 78 - 2023` directory with TXT files.

### Metadata

To reproduce the metadata preparation step run the `metadata/metadata_dataset_preparation.ipynb` and then the `metadata/metadata_text_paths_preparation.ipynb` notebooks using the main environment. It will create the `metadata/enchanced_metadata.csv` file.

### Preprocessing

To reproduce the data preprocessing step first install the `en_core_web_lg` Spacy pipeline:

```
conda run -n mining_unga_debates --no-capture-output python -m spacy download en_core_web_lg
```

Then run the appropriate script:

```
conda run -n mining_unga_debates --no-capture-output python preprocessing/main.py
```

It will create the `data_processed_add_features.pickle` file.

### Modeling

To reproduce the topic modeling step using BERTopic run:

```
conda run -n mining_unga_debates --no-capture-output python modeling/bertopic_models.py
```

It will populate the `models` directory with model weights.

### Metrics

To reproduce the metrics calculation step first you need to create a secondary Python 3.8 environment:

```
conda create -yn mining_unga_debates_metrics python=3.8 &&
conda run -n mining_unga_debates_metrics --no-capture-output pip install -r metrics/requirements.txt
```

This is due to a bug described in the following issue: [https://github.com/MIND-Lab/OCTIS/issues/114](https://github.com/MIND-Lab/OCTIS/issues/114).

Then, you need to convert the dataset to an OCTIS package compatible format:

```
conda run -n mining_unga_debates_metrics --no-capture-output python metrics/octis_dataset.py
```

It will create the `metrics/octis_dataset/corpus.tsv` file.

Finally, you can calculate the metric values:

```
conda run -n mining_unga_debates_metrics --no-capture-output python metrics/metric_values.py
```

It will create the `metrics/metric_values.csv` file.

### Analysis

To reproduce the topic analysis steps run the respective notebooks in the `analysis` directory using the main environment.
