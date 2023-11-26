import os
from SoccerNet.Downloader import SoccerNetDownloader

if __name__ == "__main__":
    """Download all videos present in SoccerNet in 224p quality (all splits)."""

    save_path = "./../video/data"
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=save_path)
    # Add password needed to download videos (NDA => password protected)
    mySoccerNetDownloader.password = os.environ.get("PASSWORD")
    mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"],
                                        split=["train", "valid", "test", "challenge"])
