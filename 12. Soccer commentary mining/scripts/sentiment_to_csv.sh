#! /bin/bash
#SBATCH --cpus-per-task 4
#SBATCH --time 04:00:00
#SBATCH --partition=experimental
#SBATCH --job-name SoccerNet_Sentiment_To_CSV
#SBATCH --account re-com
#SBATCH --mem 64G
#SBATCH --output=logs/Sentiment_To_CSV_log.txt
#SBATCH --error=logs/Sentiment_To_CSV_err.txt

. ~/.pyenv/versions/nlp/bin/activate
python3 ./src/vader_sentiment.py