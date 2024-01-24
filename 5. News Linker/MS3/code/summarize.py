import argparse
from evaluate import summarize_intersection, summarize_npmi, summarize_presence, hist_npmi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='sta', type=str, help='name of dataset folder')
    parser.add_argument('--first_topic', type=int, help='id of the first available topic')
    parser.add_argument('--last_topic', type=int, help='int of the last available')
    args = parser.parse_args()

    topics = [f"topic_{topic}" for topic in range(args.first_topic, args.last_topic + 1)]

    acc, nonzero = summarize_intersection(args.dataset, topics)
    print(f"Average accuracy: {acc}\nNumber of non-zero intersection sets: {nonzero}")

    average_presence = summarize_presence(args.dataset, topics)
    print(f"Average presence in the extended document list: {average_presence}")
    print("*\'Presence\': percentage of ground-truth documents that are present in an unfiltered list of document id predictions")

    avg_npmi, npmi_per_doc = summarize_npmi(args.dataset, topics)
    print(f"Average NPMI: {avg_npmi}")
    print("*NPMI: (-1: terms never occur together, 0: terms are independent, 1: complete co-occurrence)")
    
    hist_npmi(args.dataset, npmi_per_doc)
    
