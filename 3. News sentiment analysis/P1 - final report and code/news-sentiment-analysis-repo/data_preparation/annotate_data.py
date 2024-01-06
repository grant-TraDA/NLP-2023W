import os
import json
import pandas as pd


def prepare_ids_keywords_and_texts(directory):
    ids, keywords, texts = [], [], []
    for filename in os.listdir(directory):
        with open(directory + filename, "r") as file:
            data = json.load(file)
            for news in data:
                if "COPYRIGHT" in news["keywords"]:
                    continue
                if "text" in news:
                    ids.append(news["id"])
                    keywords.append(", ".join(news["keywords"]))
                    if "lede" in news:
                        texts.append(news["lede"] + ' ' + news["text"])
                    else:
                        texts.append(news["text"])
                elif "lede" in news:
                    ids.append(news["id"])
                    keywords.append(", ".join(news["keywords"]))
                    texts.append(news["lede"])
    return ids, keywords, texts


def create_excel_to_annotate(directory, out_file):
    ids, keywords, texts = prepare_ids_keywords_and_texts(directory)
    df = pd.DataFrame(data={
        "id": ids,
        "text": texts,
        "keywords": keywords
    })
    df["keywords_labels"] = ""
    df["full_sentiment"] = ""
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    df.to_excel(out_file, index=False)


def find_news(directory, id):
    ids, keywords, texts = prepare_ids_keywords_and_texts(directory)
    i = ids.index(id)
    print(texts[i], end="\n\n")
    print(f"Keywords: " + keywords[i])


if __name__ == '__main__':
    language = "en"
    find_news("../data_preparation/raw_data/" + language + '/', 3210455)
    # out_name = language + ".xlsx"
    # out_dir = 'annotation'
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # create_excel_to_annotate("../data_preparation/raw_data/" + language + '/', out_dir + '/' + out_name)
