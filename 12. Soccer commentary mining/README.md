# Soccer Commentary Mining

## 🗒️ Authors
|  Student's  name  |
|-------------------|
| Szymon  Maksymiuk |
|  Adam  Narozniak  |
| Władysław Olejnik |
|  Patryk  Świątek  |

## 🌳 Project structure
```bash
│   .gitignore # Files to ignore
│   README.md # This file
│   requirements.in # File used by pip-compile
│   requirements.txt # Dependency requriements to reproduce the experiment
├───data
│       labels.csv # Labels
│       sentiment.csv # Extracted sentiment
│       video_encoding.json # Video encoding
├───docs # Documents, reports, presentations
├───logs # Folder for logs
├───scripts # Bash scripts used to perform project tasks
└───src
    │   (...).py # Python files each for one project task; used by bash scripts
    │
    └───SoccerNetExplorer # Our module used to explore dataset and handle its complicated structure
        ├───audio_emotions # Submodule for extracting emotions
        ├───flair_sentiment # Submodule for extracting sentiment with Flair
        ├───labels_data # Submodule for preparing labels
        └───vader_sentiment # Submodule for extracting sentiment with Vader
```
## 📖 Reproduce Text-related Part
**Installation**

Create a virtualenv (e.g. with pyenv)
```bash
# Create
pyenv virtualenv soccernet 3.10.9
# Automatically activate
pyenv local soccernet
```
Install python requirements
```bash
python -m pip install -r ./requirements.txt
```

**Scripts**

Run project tasks
```bash
python (...).py
```

To run all project tasks run
```bash
python main.py
```
**Warning:** It may take a while!


## 🎙️ Reproduce Audio Part
**Installation**

Create a virtualenv (e.g. with pyenv)
```bash
# Create
pyenv virtualenv soccernet 3.10.9
# Automatically activate
pyenv local soccernet
```
Install python requirements
```bash
python -m pip install -r ./requirements.txt
```
Some system miss `ffmpeg` (it's valid for some audio libraries). Let's install it too:
```bash
sudo apt-get install fmpeg
```
Create a session in `tmux` and reactivate the created `venv`. The computation takes long time and to avoid the process 
termination is good to run in e.g. `tmux`.
```bash
tmux attach
pyenv activate soccernet
```

**Download video**
Note that this process may take ~1 day to download all videos. Please make sure that the process finished (e.g. use 
`tmux` as suggested above)
```bash
cd src
PASSWORD="your-password" python download_soccer_net.py  
```
**Extract audio**
```bash
python extract_audio.py
```

[optional] If you want to check the audio files metrics from command line I recommend installing `sox`:
```bash
sudo apt-get install -y sox
# Install a handler for .mp3
sudo apt-get install -y libsox-fmt-mp3
```

**Emotion recognition**
```bash
python emotion_recognition.py
```

The result of running `emotion_recognition.py` is a `.csv` file per each audio file with the following structure:


 | start_frame | stop_frame | prob_neutral | prob_angry | prob_happy | prob_sad | score | text_lab | index |
|-------------|------------|--------------|------------|------------|----------|-------|----------|-------|
| 0           | 16 000     | 0.0          | 0.0        | 0.0        | 0        | 0     | 0        | 0     |

The meaning of each them are:
* `start_frame ` and `stop_frame` are absolute number of samples of audio file (the sampling rate is 16 000 samples/s).
* (prob_neutral, prob_angry, prob_happy, prob_sad) - probability of the predicted emotions

The remaining fields are kept for convenience:
* score - the highest probability 
* text_lab - predicted label
* index - index of the list ["neutral", "angry", "happy", "sad"] that determines the predcition