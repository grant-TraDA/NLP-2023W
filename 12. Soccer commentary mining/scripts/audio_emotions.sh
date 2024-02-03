#! /bin/bash
#SBATCH --cpus-per-task 8
#SBATCH --time 24:00:00
#SBATCH --partition=experimental
#SBATCH --job-name SoccerNet_Audio_Emotions
#SBATCH --account re-com
#SBATCH --mem 128G
#SBATCH --output=logs/emotions_log.txt
#SBATCH --error=logs/emotions_err.txt

. ~/.pyenv/versions/nlp/bin/activate
python3 ./src/audio_emotions.py