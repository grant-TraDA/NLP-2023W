from pathlib import Path

from SoccerNetExplorer import Explorer

if __name__ == "__main__":
    exp = Explorer(Path.cwd() / "data")
    exp.get_flair_sentiment(n_workers=16)
