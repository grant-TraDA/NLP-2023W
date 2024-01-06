# news-sentiment-analysis

The project has following structure:
- EDA
    - contains notebook with exploratory data analysis
- aspect_based_sentiment_analysis
    - files needed for ABSA task. Additional README in this directory provides further information.
- document_based_sentiment_analysis
    - files needed for document based sentiment analysis task. Additional README in this directory provides further information.
- lib
    - is a library of functions that we use for data preprocessing that precedes both of our sentiment anlysis tasks (document and aspect based). Here, we have extracted common functionalities for those tasks
- xai_for_pretrained
    - Notebooks to obtain explanations. Additional README in this directory provides further information.
- requirements.txt - file with necessary libraries to reproduce all results
- data preparation
    - raw_data - empty folder in which you need to place the data. If you are eligible to see them, we can provide data necessary to reproduce our results.
    - moreover, it has scripts downloading the data and preparing them for annotation. Additional README in this directory provides further information.
- calculate_metrics
    - contains eval.ipynb which allows to compute confusion matrices with the use of manually assigned labels.
- visualizations
    - code for preparing visualizations using model predictions.Additional README in this directory provides further information.