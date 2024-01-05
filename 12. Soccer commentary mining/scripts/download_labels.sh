#! /bin/bash
#SBATCH --cpus-per-task 1
#SBATCH --time 01:00:00
#SBATCH --partition=experimental
#SBATCH --job-name SoccerNet_Labels
#SBATCH --account re-com
#SBATCH --mem 16G
#SBATCH --output=logs/labels_log.txt
#SBATCH --error=logs/labels_err.txt

. ~/.pyenv/versions/nlp/bin/activate
python3 ./src/download_video.py