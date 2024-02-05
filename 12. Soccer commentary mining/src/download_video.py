from SoccerNetExplorer import Explorer

if __name__ == "__main__":
    exp = Explorer("./data")
    exp.download_SoccerNet(content="videos", split="all")