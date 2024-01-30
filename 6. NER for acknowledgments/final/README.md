<h1>README</h1>

The project *NER for acknowledgements* has been prepared by: Sebastian Deręgowski, Klaudia Gruszkowska, Bartosz Jamroży and Dawid Janus.

Here's a detailed description of all contents within our project files.

In folder *data*, there are six folders:
* corpus1
* corpus2
* corpus3
* corpus4
* corpus1_silver
* corpus4_silver

Four first folders contain data provided by Nina Smyrnova. Each folder has four files: *train.csv*, *dev.csv*, *test.csv* and *corpus{n}.xlsx*, where *n* is number of corpus. .csv files are used in training process.

Additionally, there are two folders with corpora enriched by silver standard set made by us. Each of those two folders have *silver_set.txt* file, that contains the dataset. Process of creation such dataset can be found in *silverset.ipynb* file. Except for that, each of those folders have the same structure: *train.csv*, *dev.csv*, *test.csv*. *dev* and *test* are the same as in original corpora (1 and 4, respectively), and *train* contains both data from original corpus, and the one from silver standard set.

Notebook *EDA.ipynb* contains all the plots, analyses and conclusions made for POC part of the project.

Notebook *Training.ipynb* contains code for model training process. This notebook shows step by step how to load a given dataset, then define embeddings, model and train the trainer object. The models are by default save in *resources* folder. However, due to large number of models and huge size of each, we decided not to put them in this repository (as they are not needed by themselves).

All trainings history is saved in .txt files in folder *training_outputs*. If no model name is provided at the beginning of the file name, then we refer to Flair Embedding model. Otherwise, the model name is stated in the name of file, as well as number of corpus and whether the corpus was enriched with silver standard set.

File *post_trainning_analysis.ipynb* contains all the model's performance evaluation, including all the plots. It is based on the training outputs.

File *Reproducibility_appendix.pdf* contains all information regarding reproducibility of the code.

All the requirements can be found in *requirements.txt* file.
