#! /bin/bash
#SBATCH --cpus-per-task 8
#SBATCH --time 24:00:00
#SBATCH --partition=experimental
#SBATCH --job-name SoccerNet_Extract_Audio
#SBATCH --account re-com
#SBATCH --mem 128G
#SBATCH --output=logs/extract_log.txt
#SBATCH --error=logs/extract_err.txt

. ~/.pyenv/versions/nlp/bin/activate
python3 ./src/extract_audio.py