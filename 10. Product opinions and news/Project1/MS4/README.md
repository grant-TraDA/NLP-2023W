# NLP-BAMK-project

## What is it?

This repository contains the source code, data sets and documentation for project 1 for the NLP course, MiNI PW 2023.
In this project we use a Python package with wrappers for different methods of sentiment and sentiment based analysis created by our team, located on a separate repository for the clarity of the code: https://github.com/bartoszrozek/pysent.

Authors:

- Anish Gupta
- Martyna Majchrzak
- Bartosz Rożek
- Konrak Welkier

## Structure of the project

### Codes

This folder contains the notebooks used to get the results and their presentation.

- EDA.ipynb - the Exploratory Data Analysis of the used datasets
- overall_results.ipynb - the execution and evaluation of tools for overall sentiment analysis on the Amazon Electronics data subset. For each tool, the value of each metric is calculated and the results are stored in *results/results_amazon_5000.csv*.
- aspect_results.ipynb - the execution and evaluation of tools/tool combinations for aspect-based sentiment analysis on the SemEval (Laptop and Restaurants) data sets. For each tool, the value of each metric is calculated and the results are stored in *results/results_laptops_5000.csv* and *results/results_restaurants_5000.csv*
- results_presentation.ipynb - creation of the plots based on the files located in the *results* subfolder
- polish_dataset_preparation.ipynb - creation of the Polish dataset by executing the best-performing PyABSA 'extraclassifier' on the PolEmo dataset
- DistilbertTrainer.ipyn - unsuccesful fine-tuning of the DistilBERT model on the Amazon electronics data sets to perform sentiment analysis, included for completeness

### Data

This folder contains data sets which can be used to test annotators:
-  amazon_electronics.csv - Amazon Electronics subset
- Laptop_Train_v2.csv and Restuarants_Train_v2.csv - SemEval 2014 
- polemo.csv - PolEmo dataset
- polemo_labelled.csv - output PolEmo dataset with aspects and labels created as the result of this project

### External

This folder contains external files which are needed to use some external tools. Currently it contains files needed to use SentiStrength.

### Requirements.txt

This file contains the required package dependencies for this project. Python 3.11.3 was used.
