# news-sentiment-analysis

The project has following structure:
- EDA
  - contains notebook with exploratory data analysis
- aspect_based_sentiment_analysis
  - notebook presenting preliminary results of sentiment based analysis with lots of comments concerning polarity classification and NER tasks
- document_based_sentiment_analysis
  - notebook presenting preliminary results of document based analysis with comments about results and achieved performance. Comments concer limitations as well as our objectives
- lib
  - is a library of functions that we use for data preprocessing that precedes both of our sentiment anlysis tasks (document and aspect based). Here, we have extracted common functionalities for those tasks
- xai_for_pretrained
  - notebook that showcases one of XAI techiniques we intend to use. This method is called integrated gradients and will allow us to explain predicitions of out models.
- requirements.txt - file with necesary libraries to reproduce all results
- data preparation
  - raw_data - empty folder in which you need to place the data. If you are eligible to see them, we can provide data necessary to reproduce our results.
  - download_data.py - python script we used to download the data
  - annotate_data.py - python script that we used to prapare data for annotation.