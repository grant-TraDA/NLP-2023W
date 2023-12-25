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
5. Place a text file with input seeds (e.g. ```topic1.txt```) in ```datasets/sta/``` directory.
6. Run the main algorithm:
```
python main.py --dataset sta --text_file articles/article_texts.txt --pretrain_emb word_embs.txt --topic topic1 --num_iter 2
```

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
