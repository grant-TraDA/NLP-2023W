#! /bin/bash
#SBATCH --cpus-per-task 8
#SBATCH --time 24:00:00
#SBATCH --partition=experimental
#SBATCH --job-name SoccerNet_Audio_Loudness
#SBATCH --account re-com
#SBATCH --mem 128G
#SBATCH --output=logs/loudness_log.txt
#SBATCH --error=logs/loudness_err.txt

. ~/.pyenv/versions/nlp/bin/activate
python3 ./src/audio_loudness.py