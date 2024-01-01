# News Linking
The source code of News Linking project. 

## Run News Linking
1. Place the STA corpus (```articles/```,```photos/```,```videos/```) in ```datasets/sta/``` directory.
2. Place SloBERTA-based word representations (```sta_sloberta```, ```sta_sloberta.vectors.npy```) in ```datasets/sta/``` directory.
3. Place fasttext word embeddings (```word_embs.txt```) in the main directory.
4. Compile CatE by:
```
cd cate
make cate
cd ..
```
5. Prepare training and testing datasets by either running the script:
```
python prepare.py --dataset sta --test_size 50 --text_file corpus_train.txt
```
or, if you want to use an prepated dataset split, by placing ```corpus_train.txt```, ```corpus_test.json```, ```doc2enum.json```, ```enum2doc.json```, ```sentences.json```, ```npmi_word_frequency.json```,```npmi_word_frequency_in_document.json``` files and ```topics/``` subdirectory in the dataset's main directory (ex. ```datasets/sta/```)

- The script merges articles, photos and videos into a single text corpus and randomly selects a predefined number of documents (```test_size```) to be used as testing data. 
- The remaining documents constitute a training corpus, utilized to establish mappings between documents' STA IDs and their positional IDs within the training corpus. 
- Additionally, the script computes word frequency in the training corpus (required for evaluating NPMI metrics)
- Each test document is stored in the respective  ```topics/topic_N``` directory. Its text is fed to a Slovenian Named Entity Recognition (NER) pipeline, and the identified named entities are chosen as the primary input seeds.
6. Inspect the primary input seeds in ```topic_N/topic_N.txt``` file. Add / remove seeds (if needed).
7. Run the main algorithm for one topic:
```
python newslinking.py --dataset sta --text_file corpus_train.txt --pretrain_emb word_embs.txt --topic topic_N --num_iter 4
```
Note: The algorithm can be executed for one topic (one test document) at a time. Batch execution is not supported.

8. Extracted topic-related terms are placed in ```topic_N/res_topic_N.txt``` file. IDs of the most related documents are placed in ```topic_n/doc_ids_pred.txt``` file.

# SeedTopicMine
The source code used for paper "[Effective Seed-Guided Topic Discovery by Integrating Multiple Types of Contexts](https://arxiv.org/abs/2212.06002)", published in WSDM 2023.

## Data
We use two benchmmark datasets, NYT and Yelp, in our paper, adapted from [**here**](https://github.com/yumeng5/CatE/tree/master/datasets). We use 60% as training corpus and the remaining 40% for evaluation.

Use the following command to generate PLM embeddings for the training corpus (gpu required)
```
python plm_emb.py
```

## Run SeedTopicMine
Before the first run, compile CatE by 
```
cd cate
make cate
cd ..
```
Then run the following command for SeedTopicMine
```
python main.py --dataset nyt --topic locations
```


## Baselines
4 baselines are compared in our paper: SeededLDA, Anchored CorEx, KeyETM, and CatE.

To reproduce the results of SeededLDA and Anchored CorEx, please refer to ```./baselines/SeededLDA.py``` and ```./baselines/AnchoredCorEx.py```, respectively.

To reproduce the results of KeyETM and CatE, please refer to their GitHub repositories (i.e., [**KeyETM**](https://github.com/bahareharandizade/keyetm) and [**CatE**](https://github.com/yumeng5/CatE)).

## Annotations
To compute P@_k_ and NDCG@_k_ scores of SeedTopicMine and the baselines, we invite five annotators to independently judge if each discovered term is discriminatively relevant to a seed. We release the annotation results in ```./annotations/```. For example, ```./annotations/yelp_sentiment_annotation.txt``` is as follows:
```
Term	Annotator1	Annotator2	Annotator3	Annotator4	Annotator5
also	none	none	none	none	none
amazing	good	none	good	good	good
anger	bad	bad	bad	bad	bad
apathetic	bad	bad	bad	bad	bad
appalling	bad	bad	bad	bad	bad
```
There are 6 columns. The first column is the term. The other 5 columns are the relevant category of the term according to the 5 annotators, respectively. If a term is relevant to more than one category or is irrelevant to any category, the category will be marked as "none".

## Citation
If you find the implementation useful, please cite the following paper:
```
@inproceedings{zhang2023effective,
  title={Effective Seed-Guided Topic Discovery by Integrating Multiple Types of Contexts},
  author={Zhang, Yu and Zhang, Yunyi and Michalski, Martin and Jiang, Yucheng and Meng, Yu and Han, Jiawei},
  booktitle={WSDM'23},
  pages={429--437},
  year={2023}
}
```
