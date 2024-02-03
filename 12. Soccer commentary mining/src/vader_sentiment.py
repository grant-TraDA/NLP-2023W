from SoccerNetExplorer import Explorer
from pathlib import Path

if __name__ == "__main__":
    exp = Explorer(Path.cwd()/"data")
    exp.get_vader_sentiment(n_workers=4)
