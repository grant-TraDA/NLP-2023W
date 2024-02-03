import json
import os
from pathlib import Path

import pandas as pd
from SoccerNet.Downloader import SoccerNetDownloader
from tqdm import tqdm

from SoccerNetExplorer.audio_emotions import audio, emotion, loudness
from SoccerNetExplorer.flair_sentiment import flair_sentiment
from SoccerNetExplorer.vader_sentiment import vader_sentiment
from SoccerNetExplorer.labels_data import LabelsData, merge_sentiments_labels


class Explorer:
    def __init__(self, root_dir: str = "data", log_dir: str = "logs") -> None:
        """
        Initialize the SoccerNetExplorer class.

        Args:
            root_dir (str): The root directory path where the data is located. Default is "data".

        Returns:
            None
        """

        self.data_root: Path = Path(root_dir)  # Set the root directory path
        self.log_dir: Path = Path(log_dir)  # Set the log directory path
        self.audio_root: Path = (
            self.data_root / "audio"
        )  # Set the audio directory path

        self.match_folders = self.get_folders(
            self.data_root
        )  # Get the folders containing the matches
        self.videos = self.get_video_pths(
            self.data_root
        )  # Get the paths of the videos
        self.video_encodig = self.get_video_encodig()  # Get the video encoding

    def _split_label_json(self, pth: Path) -> None:
        """
        Splits the labels in the JSON file into two separate JSON files based on the game half.

        Args:
            pth (Path): The path to the directory containing the JSON file.

        Returns:
            None
        """
        with open(pth / "Labels.json", "r") as f:
            data = json.load(f)

        json1 = {"annotations": []}
        json2 = {"annotations": []}

        # Splitting the labels based on game half
        for key, value in data.items():
            if key == "annotations":
                continue
            json1[key] = value
            json2[key] = value

        for annotation in data["annotations"]:
            if str(annotation["gameTime"]).startswith("1"):
                annotation["gameTime"] = str(annotation["gameTime"])[4:]
                json1["annotations"].append(annotation)
            elif str(annotation["gameTime"]).startswith("2"):
                annotation["gameTime"] = str(annotation["gameTime"])[4:]
                json2["annotations"].append(annotation)

        # Saving the split labels into separate JSON files
        with open(pth / "labels_1.json", "w") as f:
            json.dump(json1, f)

        with open(pth / "labels_2.json", "w") as f:
            json.dump(json2, f)

    def _unpack_transcription_json(self, pth: Path, half: int):
        """
        Unpacks the transcription JSON file and saves it in a new format.

        Args:
            pth (Path): The path to the directory containing the JSON file.
            half (int): The half of the game.

        Returns:
            None
        """
        # Open the JSON file for reading
        with open(pth / f"{half}_224p_medium_asr.json", "r") as f:
            data = json.load(f)

        arr = []
        _json = {}
        # Select relevant data from the JSON file
        for segment in data.get("segments"):
            for key, value in segment.items():
                if key in ("id", "start", "end", "text"):
                    _json[key] = value
            arr.append(_json)
            _json = {}

        # Save the data to a new JSON file
        with open(pth / f"transcription_{half}.json", "w") as f:
            json.dump({"segments": arr}, f)

    def unpack_transcriptions(self, force=False, folders=None):
        """
        Unpacks transcriptions from the specified folders.

        Args:
            force (bool, optional): If True, forces the unpacking even if the transcription files already exist.
                Defaults to False.
            folders (list, optional): List of folders to unpack transcriptions from.
                If None, uses the match_folders attribute of the class. Defaults to None.
        """
        # If no folders are specified, use the match_folders attribute
        if folders is None:
            folders = self.match_folders

        # Unpack the transcriptions from the JSON files
        for pth in tqdm(folders, desc="Unpacking transcriptions"):
            if (
                force or not os.path.isfile(pth / "transcription_1.json")
            ) and os.path.isfile(pth / "1_224p_medium_asr.json"):
                self._unpack_transcription_json(pth, 1)
            if (
                force or not os.path.isfile(pth / "transcription_2.json")
            ) and os.path.isfile(pth / "2_224p_medium_asr.json"):
                self._unpack_transcription_json(pth, 2)

    def unpack_labels(self, force=False, folders=None):
        """
        Unpacks labels from JSON files in the specified folders.

        Args:
            force (bool, optional): If True, forces the unpacking even if the labels already exist. Defaults to False.
            folders (list, optional): List of folders to unpack labels from. If None, uses the match_folders attribute. Defaults to None.
        """
        # If no folders are specified, use the match_folders attribute
        if folders is None:
            folders = self.match_folders

        # Unpack the labels from the JSON files
        for folder in tqdm(folders, desc="Unpacking labels"):
            if (
                not force
                and (
                    os.path.isfile(folder / "labels_1.json")
                    or os.path.isfile(folder / "labels_2.json")
                )
            ) or not os.path.isfile(folder / "Labels.json"):
                continue
            self._split_label_json(folder)

    def get_folders(self, directory: Path) -> list[Path]:
        """
        Recursively retrieves all folders within the given directory.

        Args:
            directory (Path): The directory to search for folders.

        Returns:
            list[Path]: A list of Path objects representing the folders found.
        """
        # Initialize an empty list to store the folders
        folders = []
        # Iterate over the entries in the directory
        for entry in os.scandir(directory):
            if entry.is_dir():
                # If the entry is a directory, search for folders within it
                folders += self.get_folders(entry.path)
                # If the entry is a directory and it contains no subdirectories, add it to the list
                if all([not x.is_dir() for x in os.scandir(entry.path)]):
                    folders.append(Path(entry.path))

        return folders

    def get_video_pths(self, directory: Path) -> list[Path]:
        """
        Recursively searches for video files with the ".mkv" extension in the given directory and its subdirectories.

        Args:
            directory (Path): The directory to search in.

        Returns:
            list[Path]: A list of Path objects representing the paths to the found video files.
        """
        matches = []
        # Iterate over the entries in the directory
        for entry in os.scandir(directory):
            if entry.is_dir():
                # If the entry is a directory, search for video files within it
                matches += self.get_video_pths(entry.path)
                # If the entry is a directory and it contains no subdirectories, add it to the list
                matches += [
                    x.path
                    for x in os.scandir(entry.path)
                    if x.name.endswith(".mkv")
                ]

        return matches

    def sentiment_to_csv(self):
        """
        Converts sentiment data from JSON files to CSV format and saves them in the same directory.

        Returns:
            None
        """
        # Get the paths to the transcription JSON files
        transcriptions = [
            pth / "transcription_1.json"
            for pth in self.match_folders
            if (pth / "transcription_1.json").is_file()
        ] + [
            pth / "transcription_2.json"
            for pth in self.match_folders
            if (pth / "transcription_2.json").is_file()
        ]

        # Convert the sentiment data to CSV format
        for pth in tqdm(transcriptions, desc="Converting sentiment to csv"):
            df = self._sentiment_to_df(pth)
            name = (
                "sentiment_1.csv"
                if "transcription_1" in str(pth)
                else "sentiment_2.csv"
            )
            # Save the CSV file in the same directory as the JSON file
            df.to_csv(pth.parent / name)

    def _sentiment_to_df(self, pth: Path) -> pd.DataFrame:
        """
        Convert sentiment data from a JSON file to a pandas DataFrame.

        Args:
            pth (Path): The path to the JSON file.

        Returns:
            pd.DataFrame: The sentiment data as a pandas DataFrame.
        """
        # Load the data from the JSON file
        with open(pth, "r") as f:
            data = json.load(f)

        # Get the encoding for the video
        tmp = (
            pth.parent / "1_224p.mkv"
            if "transcription_1" in str(pth)
            else pth.parent / "2_224p.mkv"
        )
        encoding = self.video_encodig["encoding"][str(tmp)]

        tmp = []
        cols = [
            "match_id",
            "txt_id",
            "start",
            "end",
            "text",
            "flair_sentiment",
            "flair_score",
            "flair_tag",
            "vader_positive",
            "vader_negative",
            "vader_neutral",
            "vader_compound",
        ]

        # Iterate over the segments in the JSON file
        for segment in data["segments"]:
            # Append the sentiment data to the list
            tmp.append(
                [
                    encoding,
                    segment["id"],
                    segment["start"],
                    segment["end"],
                    segment["text"],
                    segment["flair"]["sentiment"],
                    segment["flair"]["score"],
                    segment["flair"]["tag"],
                    segment["vader"]["positive"],
                    segment["vader"]["negative"],
                    segment["vader"]["neutral"],
                    segment["vader"]["compound"],
                ]
            )

        # Create a pandas DataFrame from the sentiment data
        df = pd.DataFrame(tmp, columns=cols)
        return df

    def master_csv(self):
        """
        Concatenates multiple CSV files containing sentiment data into a single master CSV file.

        Returns:
            None
        """
        # Get the paths to the CSV files
        files = [
            pth / "sentiment_1.csv"
            for pth in self.match_folders
            if (pth / "sentiment_1.csv").is_file()
        ] + [
            pth / "sentiment_2.csv"
            for pth in self.match_folders
            if (pth / "sentiment_2.csv").is_file()
        ]
        # Concatenate the CSV files into a single DataFrame
        df = pd.read_csv(files[0])
        for file in tqdm(files[1:], desc="Creating master csv"):
            df_tmp = pd.read_csv(file)
            df = pd.concat([df, df_tmp])

        # Reset the index and drop the old index column
        df.reset_index(drop=True, inplace=True)
        df.drop(columns=["Unnamed: 0"], inplace=True)

        # Save the DataFrame as a CSV file
        df.to_csv(self.data_root / "sentiment.csv")

    def extract_audio(self):
        audio(self.data_root, self.audio_root, self.log_dir)

    def emotions_from_audio(self):
        emotion(self.audio_root, self.data_root / "emotion", self.log_dir)

    def loudness_from_audio(self):
        loudness(self.audio_root, self.data_root / "loudness", self.log_dir)

    def download_SoccerNet(
        self, content: str = "all", split: str = "all"
    ) -> None:
        """
        Downloads SoccerNet data based on the specified content and split.

        Args:
            content (str, optional): Specifies the content to download. Defaults to "all".
                Possible values: "all", "videos", "labels".
            split (str, optional): Specifies the split to download. Defaults to "all".
                Possible values: "all", "train", "valid", "test", "challenge".

        Returns:
            None
        """
        # Set the download path
        data_path = self.data_root
        mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=data_path)
        mySoccerNetDownloader.password = "s0cc3rn3t"

        # Download the specified content and split
        if content == "all":
            content = ["1_224p.mkv", "2_224p.mkv", "Labels.json"]
        elif content == "videos":
            content = ["1_224p.mkv", "2_224p.mkv"]
        elif content == "labels":
            content = ["Labels.json"]
        else:
            content = []

        if split == "all":
            split = ["train", "valid", "test", "challenge"]
        elif split == "train":
            split = ["train"]
        elif split == "valid":
            split = ["valid"]
        elif split == "test":
            split = ["test"]
        elif split == "challenge":
            split = ["challenge"]
        else:
            split = []

        # Download the specified content and split
        mySoccerNetDownloader.downloadGames(
            files=content,
            split=split,
        )

    def get_video_encodig(self):
        """
        Retrieves the video encoding information from a JSON file.
        If the file exists, it loads and returns the encoding information.
        If the file doesn't exist, it generates the encoding information,
        saves it to the JSON file, and returns the combined encoding and decoding information.

        Returns:
            dict: A dictionary containing the combined encoding and decoding information.
        """
        # Check if the JSON file exists
        if os.path.isfile(self.data_root / "video_encoding.json"):
            with open(self.data_root / "video_encoding.json", "r") as f:
                return json.load(f)

        encoding = {}
        decoding = {}
        # Iterate over the video paths
        for i, video in enumerate(self.videos):
            encoding[video] = i
            decoding[i] = video

        # Save the encoding information to a JSON file
        combined = {"encoding": encoding, "decoding": decoding}
        with open(self.data_root / "video_encoding.json", "w") as f:
            json.dump(combined, f)

        return combined

    def get_flair_sentiment(self, n_workers=4):
        """
        Retrieves the sentiment analysis using Flair for the transcriptions of the matches.

        Args:
            n_workers (int): The number of workers for parallel processing.

        Returns:
            None
        """
        # Get the paths to the transcription JSON files
        transcriptions = [
            pth / "transcription_1.json" for pth in self.match_folders
        ] + [pth / "transcription_2.json" for pth in self.match_folders]

        # Perform Flair sentiment analysis on the transcriptions
        results = flair_sentiment(transcriptions, n_workers=n_workers)

        # Print the results
        for r in results:
            print(r)

    def get_vader_sentiment(self, n_workers=4):
        """
        Retrieves the VADER sentiment analysis results for the transcriptions of the matches.

        Args:
            n_workers (int): The number of workers to use for parallel processing.

        Returns:
            None
        """
        # Get the paths to the transcription JSON files
        transcriptions = [
            pth / "transcription_1.json" for pth in self.match_folders
        ] + [pth / "transcription_2.json" for pth in self.match_folders]

        # Perform VADER sentiment analysis on the transcriptions
        results = vader_sentiment(transcriptions, n_workers=n_workers)

        # Print the results
        for r in results:
            print(r)


    def __load_labels(self, folders=None):
        """
        Method extracts labels data from raw .json files.

        Parameters
        ----------
        folders: list[Path] - list of the folders with target .json files
        """
        # If no folders are specified, use the match_folders attribute
        if folders is None:
            folders = self.match_folders

        objects_labels = []

        for curr_folder in folders:
            curr_object = LabelsData(curr_folder)
            try:
                curr_object.load_json()
            except:
                print(f'Labels not founded in the directory {curr_folder}')

            objects_labels.append(curr_object)

        return objects_labels
    


    def labels_to_csv(self, folders=None):
        """
        Method gathers extracts the labels data from raw .json files
        and save it to the dataframe. Then it exports the dataframe
        to csv file.

        Parameters
        ----------
        folders: list[Path] - list of the folders with target .json files
        """

        objects_labels = self.__load_labels(folders=folders)

        dict_to_df = {
            "game_video_to_encode" : [],
            "game_home_team" : [],
            "game_away_team" : [],
            "game_date_str" : [],
            "game_date_dt" : [],
            "game_score" : [],
            "url_local" : [],
            "game_video_to_encode" : [],
            "half" : [],
            "game_time" : [],
            "game_time_sec" : [],
            "label" : [],
            "position" : [],
            "visibility" : [],
            "team" : []
        }

        for obj in tqdm(objects_labels, desc="Unpacking objects"):
            labels_number = len(obj.label)
            dict_to_df['game_video_to_encode'] += obj.game_video_to_encode
            dict_to_df['game_home_team'] += [obj.game_home_team] * labels_number
            dict_to_df['game_away_team'] += [obj.game_away_team] * labels_number
            dict_to_df['game_date_str'] += [obj.game_date_str] * labels_number
            dict_to_df['game_date_dt'] += [obj.game_date_dt] * labels_number
            dict_to_df['game_score'] += [obj.game_score] * labels_number
            dict_to_df['url_local'] += [obj.url_local] * labels_number
            dict_to_df['half'] += obj.half
            dict_to_df['game_time'] += obj.game_time
            dict_to_df['game_time_sec'] += obj.game_time_sec
            dict_to_df['label'] += obj.label
            dict_to_df['position'] += obj.position
            dict_to_df['visibility'] += obj.visibility
            dict_to_df['team'] += obj.team

        df_labels = pd.DataFrame(dict_to_df)

        # Save the DataFrame as a CSV file
        df_labels.to_csv(self.data_root / "labels.csv", index=False)

        return
    

    def merge_labels_sentiments(self, sentiments_path, labels_path, encoder_path):
        """
        Function merges dataframes with sentiments data and labels data
        (both loaded from dedicated .csv files) and then, saves it as
        a .csv file. Within the merging process, the encoding-decoding .json
        file is loaded in order to extracts the matches id which is the key in
        merging process.

        Parameters
        ----------
        sentiments_path: str - path to the .csv file with the sentiments data

        labels_path: str - path to the .csv file with the labels data

        encoder_path: str - path to the .json file storing the ids of the matches
            assigned in the earlier process of data processing

        """

        merge_sentiments_labels(sentiments_path=sentiments_path, 
                                labels_path=labels_path, 
                                encoder_path=encoder_path,
                                output_path=self.data_root / "sentiments_labels_merged_v2.csv")
        
        return