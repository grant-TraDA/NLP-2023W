# Data preparation.

This directory contains 3 files important for the user:
- download_data.py - this script allows to download data from press agency. The parameters to specify are places at the top of the file.
- prepare_whole_dataset - this notebook prepares the dataframe for ML tasks. It uses functions specified in the lib directory. It combined the first paragraph of the text with the remaining text, which was stored separately, and also removes formatting.
- prepare_testset - this notebook is similar to the previous one. It also prepares data for document and abstract based analysis, however, additionally it samples articles from different categories and gets them ready for the task of labelling.

More details can be found in each of this files. 