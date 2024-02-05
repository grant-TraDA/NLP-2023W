#! /bin/bash
#SBATCH --cpus-per-task 4
#SBATCH --time 04:00:00
#SBATCH --partition=experimental
#SBATCH --job-name SoccerNet_Flair
#SBATCH --account re-com
#SBATCH --mem 64G
#SBATCH --output=logs/flair_log.txt
#SBATCH --error=logs/flair_err.txt

. ~/.pyenv/versions/nlp/bin/activate
python3 ./src/flair_sentiment.py