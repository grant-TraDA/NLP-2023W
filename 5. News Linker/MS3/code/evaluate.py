import numpy as np
import argparse
import os
import json
from utils import get_word_frequencies, pmi

def evaluate_doc_ids(args):
    # directory of a dataset (ex. sta/)
    dataset_dir = f'datasets/{args.dataset}'
    # directory of a current test document (ex. sta/topic_0/)
    topic_dir = f'datasets/{args.dataset}/topics/{args.topic}' # = dataset_dir

    doc_ids_results = {}
    with open(os.path.join(topic_dir, 'doc_ids_pred.txt'),'r') as f:
        enum_id_preds = f.readlines()
    with open(os.path.join(dataset_dir, 'enum2doc.json'),'r') as f:
        enum2doc = json.load(f)
    doc_id_preds = [enum2doc[enum_id[:-1]] for enum_id in enum_id_preds]
    with open(os.path.join(topic_dir, 'doc_ids_pred_converted.txt'),'w') as f:
        for doc_id_pred in doc_id_preds:
            f.write(f"{doc_id_pred}\n")
    with open(os.path.join(topic_dir, 'doc_ids_gt.txt'),'r') as f:
        doc_id_gt = f.readlines()
    
    return doc_ids_results

def evaluate_npmi(args):
    # directory of a dataset (ex. sta/)
    dataset_dir = f'datasets/{args.dataset}'
    # directory of a current test document (ex. sta/topic_0/)
    topic_dir = f'datasets/{args.dataset}/topics/{args.topic}' # = dataset_dir
    
    data_samples = []
    with open(os.path.join(dataset_dir, 'corpus_train.txt'),'r') as f:
        for line in f.readlines():
            data_samples.append(line[:-1])

    wf_path = os.path.join(dataset_dir, 'npmi_word_frequency.json')
    wfid_path = os.path.join(dataset_dir, 'npmi_word_frequency_in_document.json')
    if os.path.exists(wfid_path) and os.path.exists(wf_path):
        with open(wf_path,'r') as fin:
            word_frequency = json.load(fin)
        with open(wfid_path,'r') as fin:
            word_frequency_in_documents = json.load(fin)
            for word in word_frequency_in_documents:
                word_frequency_in_documents[word] = set(word_frequency_in_documents[word])
    else:
        word_frequency, word_frequency_in_documents = get_word_frequencies(data_samples)

        with open(wf_path,'w') as fout:
            json.dump(word_frequency,fout)
            print(f"Saved word frequency to {wf_path}")

        word_frequency_in_documents_list = {}
        for word in word_frequency_in_documents:
            word_frequency_in_documents_list[word] = [int(i) for i in list(word_frequency_in_documents[word])]
        with open(wfid_path,'w') as fout:
            json.dump(word_frequency_in_documents_list,fout)
            print(f"Saved word frequency in document to {wfid_path}")

    num_docs = len(data_samples)
    npmi_results = {}
    res_doc = {}
    with open(os.path.join(topic_dir, f'res_{args.topic}.txt'),'r') as fin:
        for line in fin.readlines():
            if line.startswith("Category"):
                seed = line[line.find('('):line.find(')')]
            else:
                terms = line.split()
                print(seed, " : ", terms)
                res_doc[seed] = [term.replace("\"","") for term in terms]

    for seed in res_doc:
        topic_words = res_doc[seed]
        npmi_ = pmi(topic_words, word_frequency, word_frequency_in_documents, num_docs, normalise=True)
        npmi_results[seed] = npmi_

    with open(os.path.join(topic_dir,'npmi_results.json'),'w') as fout:
        json.dump(npmi_results, fout)
    print(f"Saved NPMI results to {topic_dir}/npmi_results.json")
    return npmi_results