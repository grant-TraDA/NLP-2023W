import numpy as np
import argparse
import os
import json

def save_list(l, fname):
    with open(fname,'w') as fout:
        for elem in l:
            fout.write(f'{elem}\n')
    print(f'Saved list to {fname}')

def sort_dictionary(dictionary):
    return sorted(dictionary.items(), key=lambda x:x[1], reverse=True)

def save_sorted_dictionary(dictionary, fname):
    dictionary_sorted = sort_dictionary(dictionary)
    with open(fname, 'w') as fout:
        for key, value in dictionary_sorted:
            fout.write(f'{key},{value}\n')
    print(f'Saved dictionary to {fname}')

def rank_dictionary(dictionary):
    dictionary_sorted = sort_dictionary(dictionary)
    freq_sorted = [x[1] for x in dictionary_sorted]
    ranks_dict = {}
    for key, value in dictionary_sorted:
        rank = freq_sorted.index(value) + 1
        ranks_dict[key] = rank
    return ranks_dict

def rank_documents(args, topK=5):
    # directory of a current test document (ex. sta/topic_0/)
    topic_dir = f'datasets/{args.dataset}/topics/{args.topic}' # = dataset_dir
    with open(os.path.join(topic_dir, f'{args.topic}_seeds_doc_ids.json'),'r') as f:
        data = json.load(f)

    # for each keyword of a seed there is a list of document ids
    # {document id : 
    #    number of document's sentences that are topic-indicative given a keyword in a seed}   
        
    # frequency with which sentences from the document are chosen as indicative of the selected topic,
    # considering all keywords of a particular seed
    df_seed = {}
    # frequency with which sentences from the document are chosen as indicative of the selected topic, 
    # considering all seeds and keywords
    df_total = {}
    # count of seeds for which sentences from the document were chosen as topic-indicative  
    seeds_per_doc = {}

    # iterave over all input seeds
    for seed in data:
        print(seed) 
        
        df_seed[seed] = {}
        # iterate over all terms of the seed 
        for kw in data[seed]:
            # retrieve id's of documents with sentences used as topic-indicative given a seed and a term
            doc_ids = data[seed][kw]['doc_ids']
            # iterate over the documents
            for doc_id in doc_ids:
                # retrieve number of times document's sentences were chosen 
                # as topic-indicative in a context of the seed and the term
                df_seed_term = doc_ids[doc_id]

                # update document's sentences frequency per seed (across all seed's terms)
                if doc_id in df_seed[seed]:
                    df_seed[seed][doc_id] += df_seed_term
                else:
                    df_seed[seed][doc_id] = df_seed_term

                # update document's sentences frequency across all seeds and terms
                if doc_id in df_total:
                    df_total[doc_id] += df_seed_term
                else:
                    df_total[doc_id] = df_seed_term

                # update list of seeds for which document's sentences were used as topic-indicative
                if doc_id in seeds_per_doc:
                    if seed not in seeds_per_doc[doc_id]:
                        seeds_per_doc[doc_id].append(seed)
                else:
                    seeds_per_doc[doc_id] = [seed]
    
    # change list of seeds per document to count of seeds per documents
    for doc_id in seeds_per_doc:
        seeds_per_doc[doc_id] = len(seeds_per_doc[doc_id])
    
    # rank documents based on total frequency
    ranks_df_total = rank_dictionary(df_total)

    # rank documents based of categories count
    ranks_seeds_per_doc = rank_dictionary(seeds_per_doc)

    # calculate mean reciprocal rank (MRR)
    doc2mrr = {}
    for doc_id in ranks_df_total:
        doc2mrr[doc_id] = (1/2) * (1./ranks_df_total[doc_id] + 1./ranks_seeds_per_doc[doc_id]) 
    # sort by ranks
    mrr_sorted = sort_dictionary(doc2mrr)

    # save predicted document ids
    pred_ids = [x[0] for x in mrr_sorted[:topK]]
    save_list(pred_ids, os.path.join(topic_dir, f'doc_ids_pred.txt'))
    
    # save documents total frequency
    save_sorted_dictionary(df_total, os.path.join(topic_dir, f'doc_freq_total.txt'))

    # save documents with their category count
    save_sorted_dictionary(seeds_per_doc, os.path.join(topic_dir, f'doc_seeds_count.txt'))
    return df_seed, df_total, seeds_per_doc