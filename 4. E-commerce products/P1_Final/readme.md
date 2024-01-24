# E-Commerce Product Similarity Comparison

A system that compares the similarity between different e-commerce products. It utilizes cosine similarity between BERT embeddings as the similarity score.   
The project includes various components such as data preprocessing, model training and evaluation metrics.  
Also comes with utilities that allow for replacement of the BERT model with a DistilBERT or RoBERTa implementation.

## Installation

This project requires Python 3.10 and Jupyter Notebook to run. All other dependencies can be installed with the `installation` notebook provided in the `code` directory.  
It is recommended to install GPU-ready version of pytorch in order to speed up the training process.

## Usage

A detailed instructions on how to use the entire pipeline, as well as each of the components separately, can be found in the `demonstration` notebook provided in the `code` directory.

## Project structure

Files in the `code` directory are structured into subfolders to ease of navigation. Contents of each folder are as follows:

- **preprocessing**: contains files for running data preprocessing (attribute extraction), and for loading the preprocessed dataset
- **training**: contains implementations of the pretrained models, as well as the class responsible for the full training pipeline
- **evaluation**: contains metrics for evaluating the model, both as a loss function during training (triplet loss) and the entire model after training (hierarchial metric)
- **comparison**: contains helper class for the calculation of a similarity score with a trained model based on input data
- **tests**: contains various helper functions for visual representation and verification of proper functioning of the pipeline