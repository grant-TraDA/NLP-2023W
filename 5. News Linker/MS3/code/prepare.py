import os
import json
import spacy
import argparse
import numpy as np
from utils import process_sentences, get_word_frequencies

NUM_RELATED = 10

def get_ids_with_related(data):
    ids = []
    related = []
    for id in data:
        if data[id]['related'] is not None:
            ids.append(id)
            related.append(data[id]['related'])
    return ids, related

def get_test_samples_ids(ids, size):
    if size < 1:
        size = int(len(ids) * ids)
    idx = np.random.choice(range(len(ids)),size,replace=False).astype(int)
    ids_selected = np.array(ids)[idx]
    return sorted(ids_selected.tolist())#, sorted(ids_other.tolist())

def construct_train_test_sets(args):
    dataset = args.dataset 
    test_size = args.test_size if args.test_size is not None else args.test_ratio 
    # ensemble articles, photos and videos
    data = {}
    for data_type in ['articles','photos','videos']:
        with open(f'./datasets/{dataset}/{data_type}/{data_type}.json','r') as fin:
            data = data | json.load(fin)
    print(f"Loaded {len(data)} documents")

    # get ids of samples with a non-empty list of related documents
    ids, related = get_ids_with_related(data)
    # get number of related documents
    lengths = [len(x) for x in related]
    # find those documents with 10 related ones
    mask = np.array(lengths) == NUM_RELATED
    # reduce test candidates to only those with 10 related documents
    ids = np.array(ids)[mask].tolist()
    assert len(ids) > test_size

    # select test samples from the list of documents with non-empty related list
    test_ids = get_test_samples_ids(ids, test_size)
    # training corpus = remaining samples + documents with empty related list 
    train_ids = list(set(list(data.keys())) - set(test_ids))
    # sort the training corpus by document id
    train_ids = [str(id_i) for id_i in sorted([int(id) for id in train_ids])]
    assert len(test_ids) + len(train_ids) == len(data)
    
    # each training document has 2 ids:
    # - STA ID 
    # - positional ID (id of document's line within the training corpus file
    
    # dictionary STA ID -> line ID
    doc2enum = {}
    # dictionary line ID -> STA ID
    enum2doc = {}
    # save training corpus in a format of 1 document per 1 line
    with open(f'./datasets/{dataset}/corpus_train.txt','w') as fout:
        for i, doc_id in enumerate(train_ids):
            val = data[doc_id]
            # write document's text to a new line
            fout.write(val['text'] + '\n' if val['text'] is not None else "None\n")
            # save the corresponding ids
            doc2enum[doc_id] = i
            enum2doc[i] = doc_id
        print(f"Saved training corpus to ./datasets/{dataset}/corpus_train.txt")

    # save id dictionaries
    with open(f'./datasets/{dataset}/doc2enum.json','w') as fout:
        json.dump(doc2enum, fout)
        print(f"Saved doc2enum dictionary to ./datasets/{dataset}/doc2enum.json")
    with open(f'./datasets/{dataset}/enum2doc.json','w') as fout:
        json.dump(enum2doc, fout)
        print(f"Saved enum2doc dictionary to ./datasets/{dataset}/enum2doc.json")

    # create a dictionary of sentences from a training corpus 
    if not os.path.exists(f'datasets/{args.dataset}/sentences.json'):
        process_sentences(args)

    if not os.path.exists(f'datasets/{args.dataset}/topics'):
        os.makedirs(f'datasets/{args.dataset}/topics')
    # dictionary to represent the whole test corpus
    test_data = {}
    # list of test documents' texts to be used in spaCy NER pipeline
    test_texts = []
    for i, test_id in enumerate(test_ids):
        val = data[test_id]
        rel = val['related']
        txt = val['text']
        test_data[test_id] = {'related': rel, 'text': txt}
        test_texts.append(txt)

        # for each test document create a separate sub-directory
        topic_dir = f'./datasets/{dataset}/topics/topic_{i}'
        if not os.path.exists(topic_dir):
            os.makedirs(topic_dir)

        # save ground-truth values of related document ids
        with open(os.path.join(topic_dir, "doc_ids_gt.txt"),'w') as fout:
            for doc_id in rel:
                fout.write(f"{doc_id}\n")

        # save document's text
        with open(os.path.join(topic_dir, "doc.txt"),'w') as fout:
            fout.write(txt + '\n')
    
    # save the whole test corpus in a single json file
    with open(f'./datasets/{dataset}/corpus_test.json','w') as fout:
        json.dump(test_data, fout)
        print(f"Saved test set to ./datasets/{dataset}/corpus_test.json")

    # load the slovenian nlp model
    nlp = spacy.load("sl_core_news_sm")
    # perform NER on each document to retrieve potential seeds 
    for i, doc in enumerate(nlp.pipe(test_texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])):
        topic_dir = f'./datasets/{dataset}/topics/topic_{i}'
        # save retrieved entities in a format of (entity, label)
        with open(os.path.join(topic_dir,"ner_seeds.txt"),'w') as fout:
            for ent in doc.ents:
                fout.write(f"{ent.text},{ent.label_}\n")
        # create a file with the potential input seeds
        with open(os.path.join(topic_dir,f"topic_{i}.txt"),'w') as fout:
            for ent in doc.ents:
                fout.write(f"{ent.text}\n")

    npmi = True
    if npmi:
        # for npmi calculations
        data_samples = []
        dataset_dir = f'./datasets/{dataset}'
        with open(os.path.join(dataset_dir, 'corpus_train.txt'),'r') as f:
            for line in f.readlines():
                data_samples.append(line[:-1])

        wf_path = os.path.join(dataset_dir, 'npmi_word_frequency.json')
        wfid_path = os.path.join(dataset_dir, 'npmi_word_frequency_in_document.json')
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


    print("Each test document has a separate subdirectory with the files containing:\n  \
          - its text \
          - a list of ground-truth IDs of the related documents \
          - potential input seeds extracted by NER pipeline"
          )
    
    print("NOTE 1. Before running the main algorithm, one must manually check a file \
          with the entities retrieved by NER and remove incorrect/add new seeds")
    
    print("NOTE 2. The main algorithm is run on one test document per time. \
          The corresponding topic directory \'topicN\' should be given as an input")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='sta', type=str, help='name of dataset folder')
    parser.add_argument('--test_size', type=int, help='number of documents in a test set')
    parser.add_argument('--test_ratio', type=float, help='percentage of test split')
    parser.add_argument('--text_file', default='corpus_train.txt', type=str, help='training corpus')
    args = parser.parse_args()

    if args.test_size is None and args.test_ratio is None:
        parser.error("at least one of --test_size and --test_ratio required")
    else:
        construct_train_test_sets(args)
