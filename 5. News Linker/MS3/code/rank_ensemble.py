import json
from collections import defaultdict
import math
from nltk.corpus import stopwords
import numpy as np
import argparse
from utils import *


def rank_ensemble(args, topk=20):

    word2emb = load_cate_emb(f'datasets/{args.dataset}/emb_{args.topic}_w.txt')
    #word2bert = load_bert_emb(f'datasets/{args.dataset}/{args.dataset}_bert')
    word2bert = load_bert_emb(f'datasets/{args.dataset}/{args.dataset}_sloberta')

    caseolap_results = []
    with open(f'datasets/{args.dataset}/intermediate_2.txt') as fin:
        for line in fin:
            data = line.strip()
            #_, res = data.split(':')
            data = data.split(':')
            _, res = data[0], "".join(data[1:])
            caseolap_results.append(res.split(','))
            
    with open(f'datasets/{args.dataset}/intermediate_2_doc_ids.json', 'r') as fin:
        caseolap_dict = json.load(fin)
       
    cur_seeds = []
    with open(f'datasets/{args.dataset}/{args.topic}_seeds.txt') as fin:
        for line in fin:
            data = line.strip().split(' ')
            cur_seeds.append(data)


    final_dict = {}
    with open(f'datasets/{args.dataset}/{args.topic}_seeds.txt', 'w') as fout:
        for idx, comb in enumerate(zip(cur_seeds, caseolap_results)):
            seeds, caseolap_res = comb
            word2mrr = defaultdict(float)

            # cate mrr
            word2cate_score = {word:np.mean([np.dot(word2emb[word], word2emb[s]) for s in seeds]) for word in word2emb}
            r = 1.
            for w in sorted(word2cate_score.keys(), key=lambda x: word2cate_score[x], reverse=True)[:topk]:
                if w not in word2bert: continue
                word2mrr[w] += 1./r
                r += 1
                 
            # bert mrr
            word2bert_score = {word:np.mean([np.dot(word2bert[word], word2bert[s]) for s in seeds]) for word in word2bert}
            r = 1.
            for w in sorted(word2bert_score.keys(), key=lambda x: word2bert_score[x], reverse=True)[:topk]:
                if w not in word2emb: continue
                word2mrr[w] += 1./r
                r += 1
            
            # caseolap mrr
            r = 1.
            for w in caseolap_res[:topk]:
                word2mrr[w] += 1./r
                r += 1

            score_sorted = sorted(word2mrr.items(), key=lambda x: x[1], reverse=True)
            top_terms = [x[0].replace(' ', '') for x in score_sorted if x[1] > args.rank_ens and x[0] != '']
            top_mrr = [x[1] for x in score_sorted if x[1] > args.rank_ens and x[0] != '']
            fout.write(' '.join(top_terms).replace(":","") + '\n')

            cur_dict = caseolap_dict[seeds[0]]
            terms_docs_dict = dict()
            #for term in top_terms:
            #    terms_docs_dict[term] = [] if not term in cur_dict else cur_dict[term]
            for term, mrr in zip(top_terms,top_mrr):
                docs_ids = [] if not term in cur_dict else cur_dict[term]
                terms_docs_dict[term] = {
                    'mrr': mrr, 
                    'similarity_score': docs_ids['similarity_score'] if len(docs_ids) > 0 else None,
                    'doc_ids': docs_ids['doc_ids'] if len(docs_ids) > 0 else []
                }
            final_dict[seeds[0]] = terms_docs_dict
        print(f'Saved ranked terms to datasets/{args.dataset}/{args.topic}_seeds.txt')
    with open(f'datasets/{args.dataset}/{args.topic}_seeds_doc_ids.json', 'w') as fout2:
        json.dump(final_dict,fout2)
        print(f'Saved document ids featuring ranked terms to datasets/{args.dataset}/{args.topic}_seeds_doc_ids.json')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='nyt', type=str)
    parser.add_argument('--topic', default='topic', type=str)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--rank_ens', default=0.3, type=float)
    args = parser.parse_args()

    rank_ensemble(args, args.topk)