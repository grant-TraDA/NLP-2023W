from SoccerNetExplorer import Explorer

if __name__ == "__main__":
    exp = Explorer("./data")
    exp.unpack_transcriptions()
    exp.unpack_labels()