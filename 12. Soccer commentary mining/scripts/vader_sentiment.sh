#! /bin/bash
#SBATCH --cpus-per-task 4
#SBATCH --time 04:00:00
#SBATCH --partition=experimental
#SBATCH --job-name SoccerNet_Vader
#SBATCH --account re-com
#SBATCH --mem 64G
#SBATCH --output=logs/vader_log.txt
#SBATCH --error=logs/vader_err.txt

. ~/.pyenv/versions/nlp/bin/activate
python3 ./src/vader_sentiment.py