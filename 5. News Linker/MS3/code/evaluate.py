import numpy as np
import argparse
import os
import json
from utils import get_word_frequencies, pmi
import matplotlib.pyplot as plt 


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
    
    intersection = len(set.intersection(set(doc_id_gt),set(doc_id_pred))) / len(doc_id_gt)
    with open(os.path.join(topic_dir, 'accuracy.txt'),'w') as f:
        f.write(f"{intersection}")
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

def summarize_intersection(dataset, topics):
    results = []
    for topic in topics:
        with open(f'./datasets/{dataset}/topics/{topic}/accuracy.txt','r') as fin:
            results.append(float(fin.readlines()[0]))
    results = np.array(results)
    nonzero = sum(results > 0)
    acc = results.mean()
    return acc, nonzero

def summarize_npmi(dataset, topics):
    results = []
    for topic in topics:
        with open(f'./datasets/{dataset}/topics/{topic}/npmi_results.json','r') as fin:
            res = json.load(fin)
            results.append(res)
    npmis = [list(res.values()) for res in results]
    npmi_per_doc = [sum(res) / len(res) for res in npmis]
    avg_npmi = sum(npmi_per_doc) / len(npmi_per_doc)
    return avg_npmi, npmi_per_doc
    

def hist_npmi(dataset, npmi_per_doc):
    avg_npmi = sum(npmi_per_doc) / len(npmi_per_doc)
    
    cnt,_,_ = plt.hist(npmi_per_doc)
    ymax = cnt.max()
    
    plt.vlines(0.0,0,ymax,'r')
    plt.vlines(1.0,0,ymax,'r')
    plt.vlines(-1.0,0,ymax,'r')
    plt.vlines(avg_npmi,0,ymax,'y')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(-0.2,ymax+.25,"independent",fontsize=11,verticalalignment='top', bbox=props)
    plt.text(-0.95,ymax+.25,"never occur\n   together",fontsize=11,verticalalignment='top', bbox=props)
    plt.text(0.6,ymax+.25," complete\noccurrence",fontsize=11,verticalalignment='top', bbox=props)
    
    plt.title(f"NMPI (Test set, {len(npmi_per_doc)} samples)")
    plt.xlim([-1.1,1.1])
    plt.grid(True)
    fname = f'./datasets/{dataset}/npmi_hist.png'
    plt.savefig(fname)
    print("NPMI histogram was saved to ", fname)
   

def get_all_docs(dataset, topic):
    with open(f'./datasets/{dataset}/topics/{topic}/{topic}_seeds_doc_ids.json','r') as f:
        data = json.load(f)
    results = set()
    for seed in data:
        val = data[seed]
        for term in val:
            doc_ids = val[term]['doc_ids']
            if len(doc_ids) > 0:
                results = results |set(list(doc_ids.keys()))
    return results

        
def summarize_presence(dataset, topics):
    dataset_dir = f'./datasets/{dataset}'
    with open(os.path.join(dataset_dir, 'doc2enum.json'),'r') as f:
            doc2enum = json.load(f)
    results = []
    avg_presence = []
    for topic in topics:
        # get an extended list of seed-related documents
        all_docs = get_all_docs('sta', topic)
        topic_dir = f'./datasets/sta/topics/{topic}'
        # get a list of ground-truth document STA ids
        with open(os.path.join(topic_dir, 'doc_ids_gt.txt'),'r') as f:
            doc_id_gt = [d[:-1] for d in f.readlines()]
        # number of g-t documents available in the training corpus
        num_in_corpus = len(doc_id_gt)
        # number of g-t documents present in the extended list of documents
        num_present = 0
        for d in doc_id_gt:
            if d in doc2enum:
                if str(doc2enum[d]) in all_docs:
                    num_present += 1
            else:
                num_in_corpus -= 1
        presence =  num_present / num_in_corpus if num_in_corpus > 0 else -1
        if presence > -1:
            avg_presence.append(presence)
        results.append([topic, num_in_corpus / len(doc_id_gt), presence])
           
    avg_presence = sum(avg_presence) / len(avg_presence)
    with open(f'./datasets/{dataset}/presence.txt','w') as fout:
        fout.write('topic,percentage of g-t documents present in the corpus,percentage of g-t document present in the extended list of documents')
        for result in results:
            fout.write(",".join([str(r) for r in result]) + '\n')
        fout.write(f"Average presence: {avg_presence}")
    
    return avg_presence