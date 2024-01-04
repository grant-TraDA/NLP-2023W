import os
import json
from collections import defaultdict
import math
from nltk.corpus import stopwords
import numpy as np
import argparse
from utils import *


def BM25(df, maxdf, tf, dl, avgdl, k=1.2, b=0.5):
    score = tf * (k + 1) / (tf + k * (1 - b + b * (dl / avgdl)))
    df_factor = math.log(1 + df, 2) / math.log(1 + maxdf, 2)
    score *= df_factor
    return score


def Softmax(score_list):
    exp_sum = 1
    for score in score_list:
        exp_sum += math.exp(score)
    exp_list = [math.exp(x) / exp_sum for x in score_list]
    return exp_list


def sentence_retrieval(args, seeds, keywords):
    # directory of a dataset (ex. sta/)
    dataset_dir = f'datasets/{args.dataset}'
    # directory of a current test document (ex. sta/topic_0/)
    topic_dir = f'datasets/{args.dataset}/topics/{args.topic}' # = dataset_dir

    # dictionary {seed: {sentence id: number of topic-indicative terms in the sentence}}
    scores = defaultdict(dict)
    # dictionary {sentence id: sentence text}
    id2sent = {}
    # dictionary {sentence id: id of the first sentence of the document}
    id2start = {}
    # dictionary {sentence id: id of the last sentence of the document}
    id2end = {}
    # dictionary {sentence id: document id}
    id2doc = {}
    with open(os.path.join(dataset_dir, 'sentences.json')) as fin:
        # iterate over each document (1 line = 1 document)
        for idx, line in enumerate(fin):
            if idx % 10000 == 0:
                print(idx)
            data = json.loads(line)
            start = len(id2sent)
            end = start+len(data['sentences'])-1
            # iterate over each sentence in the document
            for sent in data['sentences']:
                sent_id = len(id2sent)
                # save id of the sentence
                id2sent[sent_id] = sent
                # save ids of the first and the last sentences of the document 
                id2start[sent_id] = start
                id2end[sent_id] = end
                # save id of the sentence's document
                id2doc[sent_id] = idx

                # calculate word count in the sentence
                words = sent.split()
                word_cnt = defaultdict(int)
                for word in words:
                    word_cnt[word] += 1
                
                
                score = defaultdict(int)
                for seed in keywords:
                    # save total number of occurences of the seed's topic-indicative terms in the sentence
                    for kw in keywords[seed]:
                        score[seed] += word_cnt[kw]
                # if the sentence contains topic-indicative terms of a seed - save the seed
                pos_seeds = [x for x in score if score[x] > 0]
                # consider only sentences with topic-indicative terms of a single seed 
                if len(pos_seeds) == 1:
                    seed = pos_seeds[0]
                    # set current sentence as the 'anchor' sentence of the given seed
                    scores[seed][sent_id] = score[seed]


    # print out top sentences
    topk = args.num_sent
    wd = args.sent_window
    top_sentences = []
    with open(os.path.join(topic_dir,'top_sentences.json'), 'w') as fout:
        # iterate over each input seed
        for seed in seeds:
            out = {}
            out['seed'] = seed
            out['sentences'] = []
            # sort the anchor sentences based on the number of topic-indicative (t-i) terms they contain
            scores_sorted = sorted(scores[seed].items(), key=lambda x: x[1], reverse=True)
            # select the top-k best scored sentences
            scores_sorted = scores_sorted[:topk]
            # print(scores_sorted[-1])

            # iterate over each selected sentence
            for k0, v in scores_sorted:
                #out['sentences'].append(id2sent[k0])
                # start with the 'anchor' sentences - the ones with the maximal number of t-i terms
                out['sentences'].append({
                    'doc_id': id2doc[k0], 
                    'score': v ,
                    'sentence': id2sent[k0]
                })
                # check the preceeding neighbours of the 'anchor' sentence
                for k in range(k0-1, k0-wd-1, -1):
                    # stop when approaching end of the previous document
                    if k < id2start[k0]:
                        break
                    excl = 1
                    # check whether the sentence contains t-i terms from other seeds
                    for seed_other in seeds:
                        if seed_other == seed:
                            continue
                        # if such term was found - do not check preceeding sentences 
                        if k in scores[seed_other]:
                            excl = 0
                            break
                    if excl == 1:
                        # no terms from other seeds were found - add as 'neighbour'
                        # Note: it's possible that the neighbour doesn't have terms of the given seed
                        out['sentences'].append({
                            'doc_id': id2doc[k],
                            'score': scores[seed][k] if k in scores[seed] else 0,
                            'sentence': id2sent[k]
                        })
                    else:
                        break
                
                # check the succeeding neighbours of the 'anchor' sentence
                for k in range(k0+1, k0+wd+1):
                    # stop when approaching start of the next document
                    if k > id2end[k0]:
                        break
                    excl = 1
                    # check whether the sentence contains t-i terms from other seeds
                    for seed_other in seeds:
                        if seed_other == seed:
                            continue
                        # if such term was found - do not check succeeding sentences
                        if k in scores[seed_other]:
                            excl = 0
                            break
                    if excl == 1:
                        # no terms from other seeds were found - add as 'neighbour'
                        # Note: it's possible that the neighbour doesn't have terms of the given seed
                        out['sentences'].append({
                            'doc_id': id2doc[k],
                            'score': scores[seed][k] if k in scores[seed] else 0,
                            'sentence': id2sent[k]
                        })
                    else:
                        break
            fout.write(json.dumps(out)+'\n')
            top_sentences.append(out)
    return top_sentences

def caseolap(args, topk=20):
    # directory of a dataset (ex. sta/)
    dataset_dir = f'datasets/{args.dataset}'
    # directory of a current test document (ex. sta/topic_0/)
    topic_dir = f'datasets/{args.dataset}/topics/{args.topic}' # = dataset_dir
    seeds = []
    keywords = {}
    with open(os.path.join(topic_dir, f'intermediate_1.txt')) as fin:
        for line in fin:
            data = line.strip().split(':')
            seed = data[0]
            other = ''.join(data[1:])
            seeds.append(seed)
            #kws = [data[0]] + data[1].split(',')
            kws = [data[0]] + other.split(',')
            keywords[seed] = kws

    # learned seed-guided text embeddings of the input corpus
    word2emb = load_cate_emb(os.path.join(topic_dir, f'emb_{args.topic}_w.txt'))
    # PLM-based (SloBERTa-based) representations of the most popular slovenian words
    word2bert = load_bert_emb(os.path.join(dataset_dir, f'{args.dataset}_sloberta'))

    # for each seed retrieve topic-indicative sentences (TIS)
    top_sentences = sentence_retrieval(args, seeds, keywords)

    # number of input seeds/topics/categories
    n = len(seeds)
    # term frequency: number of times the word appears in the seed's TISes
    tf = [defaultdict(int) for _ in range(n)]
    # document frequency: number of seed's TISes where the current word appears 
    # document=all TISes of a seed
    df = [defaultdict(int) for _ in range(n)]

    # dictionary {seed: {word: [ids of documents where the word appears in context of the seed]}}
    word2doc = []
    # iterate over each seed
    for idx, data in enumerate(top_sentences):
        word2doc.append(dict())
        # iterate over each TIS retrieved the seed
        for sent in data['sentences']:
            # id of the document where the TIS appears
            doc_id = sent['doc_id']
            # count to the seed's topic-indicative terms within the sentence
            tit_count = sent['score']
            # full text of the TIS
            sent = sent['sentence']
            # words of the TIS
            words = sent.split()
            # iterate over each word of the TIS
            for word in words:
                # update the number of times a word appears in the seed's TISes
                tf[idx][word] += 1

                # save an id of the document where the word appeared in the seed's context
                if word in word2doc[idx]:
                    word2doc[idx][word].append(doc_id)
                else:
                    word2doc[idx][word] = [doc_id]

            # get the unique words of the current TIS
            words = set(words)
            # update the count of TISes where the word appeard in the current seed's context
            for word in words:
                df[idx][word] += 1

    stop_words = set(stopwords.words('slovene'))
    # a set of unique candidate words (regardless of the seed)
    candidate = set()
    # dictionary: {word: {seed id: [ids of documents where the word appeared in seed's context]}}
    candidate_states = dict()
    # iterate over each seed
    for idx in range(n):
        # iterate over each word appearing in the seed's TISes
        for word in tf[idx]:
            # proceed with non-stop words appearing at least 5 times across the seed's TISes
            if tf[idx][word] >= 5 and word not in stop_words:
                # save the word to a list of candidates
                candidate.add(word)
                # save id of the document where the candidate word appeared in the seed's context
                if word in candidate_states:
                    candidate_states[word][idx] = word2doc[idx][word]
                else:
                    candidate_states[word] = {idx : word2doc[idx][word]}
                
    # for each seed find the largest number of seed's TISes a word appears in
    maxdf = [max(df[x].values()) for x in range(n)]
    # length of seed's TISes in words (terms)
    dl = [sum(tf[x].values()) for x in range(n)]
    # average number of words in the seed's TISes over all seeds' TISes
    avgdl = sum(dl) / len(dl)
    # relevance between a term and seed's TISes
    bm25 = [defaultdict(float) for _ in range(n)]
    # iterate over each seed
    for idx in range(n):
        # iterate over each term candidate
        for word in candidate:
            # find relevance between a word and TISes of the current seed
            bm25[idx][word] = BM25(df[idx][word], maxdf[idx], tf[idx][word], dl[idx], avgdl)
    # calculate distinctiveness between a word and TISes of the current seed
    dist = {}
    for word in candidate:
        dist[word] = Softmax([bm25[x][word] for x in range(n)])

    total_docs = dict()
    with open(os.path.join(topic_dir, f'intermediate_2.txt'), 'w') as fout1:
        # for each seed find the top-scored terms based on 3 contexts
        for idx in range(n):
            seed = seeds[idx]
            caseolap = {}
            # calculate similarity score between a candidate term and the seed
            for word in candidate:
                if word in word2emb and word in word2bert:
                    # Context 1: Seed-guided text embeddings
                    # similarity between the learned embeddings of the seed and the candidate term 
                    sim1 = np.dot(word2emb[word], word2emb[seed])

                    # Context 2: PLM-based represetations
                    # similarity between representations of the seed and the candidate term
                    sim2 = np.dot(word2bert[word], word2bert[seed])

                    # Context 3: Topic-indicative sentence (TIS)
                    # popularity: how often the candidate term appears in the TIS
                    popularity = math.log(1 + df[idx][word], 2)
                    # distinctiveness: how unique is the candidate term to the TIS compared other topics' TISes
                    distinctiveness = dist[word][idx]
                    # similarity of a candidate term and category topic based on its TIS
                    sim3 = (popularity ** args.alpha) * (distinctiveness ** (1-args.alpha))

                    # final similariy of a candidate term and category topic - all contexts ensembled
                    caseolap[word] = sim3 * sim2 * sim1  
            # sort candidate words based on their similarity scores with the seed   
            caseolap_sorted = sorted(caseolap.items(), key=lambda x: x[1], reverse=True)
            # retrieve k terms with the highest simirality scores
            top_terms = [x[0] for x in caseolap_sorted[:topk]]
            # save the retrieved terms
            fout1.write(seed+':'+','.join(top_terms).replace(":","")+'\n')

            # get a list of topic indicative sentences for a given seed
            cur_sntns = top_sentences[idx]["sentences"]
            terms_docs_dict = dict()
            # iterate over each topic-indicative term of the seed
            for term, sim_score in zip(top_terms, caseolap_sorted[:topk]):
                # {document id: count of the term's occurences in the seed's t-i sentences with the document}
                docs_ids = {}
                # iterate over each topic-indicative sentence
                for sntn in cur_sntns:
                    # count occurences of the term in the sentence
                    occs = sntn['sentence'].count(term)
                    # if the term occurs - save the id of the document where the sentence is located 
                    if occs > 0:
                        if sntn['doc_id'] in docs_ids:
                            docs_ids[sntn['doc_id']] += occs
                        else:
                            docs_ids[sntn['doc_id']] = occs
                # save the term's similarity score together with the document's ids and the term's counts
                terms_docs_dict[term] = {'similarity_score': sim_score[1], 'doc_ids': docs_ids}
            # save a dictionary of seed's t-i terms and the corresponding scores and documents
            total_docs[seed] = terms_docs_dict
        print(f'Saved top-{topk} terms to {topic_dir}/intermediate_2.txt')
        
    with open(os.path.join(topic_dir, f'intermediate_2_doc_ids.json'), 'w') as fout2:
        json.dump(total_docs, fout2)
        print(f'Saved document ids featuring top-{topk} terms to {topic_dir}/intermediate_2_doc_ids.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='nyt', type=str)
    parser.add_argument('--topic', default='topic', type=str)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--num_sent', default=500, type=int)
    parser.add_argument('--sent_window', default=4, type=int)
    args = parser.parse_args()
    
    caseolap(args, args.topk)