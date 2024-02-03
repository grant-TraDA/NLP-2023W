from SoccerNetExplorer import Explorer

if __name__ == "__main__":
    exp = Explorer("./data")
    exp.download_SoccerNet(content="videos", split="all")
    exp.download_SoccerNet(content="labels", split="all")
    exp.unpack_transcriptions()
    exp.unpack_labels()
    exp.labels_to_csv()
    exp.get_flair_sentiment(n_workers=16)
    exp.get_vader_sentiment(n_workers=16)
    exp.sentiment_to_csv()
    exp.master_csv()
    exp.merge_labels_sentiments()
