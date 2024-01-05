#! /bin/bash
#SBATCH --cpus-per-task 1
#SBATCH --time 24:00:00
#SBATCH --partition=experimental
#SBATCH --job-name SoccerNet_Download
#SBATCH --account re-com
#SBATCH --mem 16G
#SBATCH --output=logs/download_log.txt
#SBATCH --error=logs/download_err.txt

. ~/.pyenv/versions/nlp/bin/activate
python3 ./src/download_video.py