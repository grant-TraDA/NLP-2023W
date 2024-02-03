import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


class LabelsData:
    """
    Class for gathering the information from the .json file with labels
    from one 'Labels-v2.json' downloaded from the original SoccerNet data.
    It contains labels storing the information about important parts of
    the specific match.

    Attributes
    ----------
    path: str - path to the file (given as parameter)

    file_content: dict - raw data loaded from the original .json file

    game_home_team: str - the team playing at home

    game_away_team - the team playing away

    game_date_str: str - datetime of the match saved as string

    game_date_str: datetime.datetime - datetime of the match saved as
        proper datetime python format

    game_score: str - final score of the match

    url_local: str - path to the local directory of the file

    game_video_to_encode: list[str] - list of paths to the files storing
        match broadcats (by default it consists of repeating two unique
        elements two unique elements, each for one half of the match broadcast).
        It is used for assigning the id to the observations.

    half: list[str] - stores information about the half of the match in
        which the specific action happened (only '1' and '2' elements
        are possible)

    game_time: list[str] - stores information about the match time when
        the specific action happened (in format 'mm:ss')

    game_time_sec: list[float] - the seconds of the match time when
        the specific actions happened (in format 'mm:ss')

    label: list[str] - list of labels indicating the type of the action
        (for SoccerNet-v2 dataset, there are 17 types of action labelled)

    visibility: list[str] - information wherther the action was visible
        during the match broadcast ('visible') or not ('not shown')

    team: list[str] - information about which team the action concerned

    """
    def __init__(self, path):
        self.path = path
        self.file_content = None
        self.game_home_team = None
        self.game_away_team = None
        self.game_date_str = None
        self.game_date_dt = None
        self.game_score = None
        self.url_local = None

        self.game_video_to_encode = [] # for merging with sentiments data
        self.half = []
        self.game_time = []
        self.game_time_sec = []
        self.label = []
        self.position = []
        self.visibility = []
        self.team = []


    def load_json(self):
        """
        Method loads the .json file containing the information
        about the labels and extracts features from the file
        conent.
        """
        if not os.path.isfile(self.path / "Labels-v2.json"):
            raise Exception('No Labels-v2.json file!!!')
        with open(os.path.join(self.path,"Labels-v2.json")) as f:
            self.file_content = json.load(f)
            self.__extract_features()


    def __extract_features(self):
        """
        Method for extracting features from .json file content.
        """
        self.game_home_team = self.file_content['gameHomeTeam']
        self.game_away_team = self.file_content['gameAwayTeam']
        self.game_score = self.file_content['gameScore']

        #match date/time
        game_date_str = self.file_content['gameDate']
        self.game_date_str = game_date_str
        self.game_date_dt = datetime(day=int(game_date_str.split('/')[0]),
                                    month=int(game_date_str.split('/')[1]),
                                    year=int(game_date_str.split('/')[2].split(' ')[0]),
                                    hour=int(game_date_str.split('/')[2].split(' ')[-1][0:2]),
                                    minute=int(game_date_str.split('/')[2].split(' ')[-1][-2:]))

        self.url_local = self.file_content['UrlLocal']


        for annot in self.file_content['annotations']:
            curr_half = annot["gameTime"][0]
            self.half.append(curr_half)
            self.game_video_to_encode.append(f'data/{self.url_local}{curr_half}_224p.mkv')

            curr_game_time = str(annot["gameTime"])[4:]
            self.game_time.append(curr_game_time)
            self.game_time_sec.append(int(curr_game_time[0:2])*60+int(curr_game_time[-2:]))

            self.label.append(annot['label'])
            self.position.append(annot['position'])
            self.team.append(annot['team'])
            self.visibility.append(annot['visibility'])
