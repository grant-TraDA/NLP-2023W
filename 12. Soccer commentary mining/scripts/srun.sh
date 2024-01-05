#! /bin/bash
srun --pty -A re-com --time 1:00:00 --cpus-per-task=4 -p experimental --mem=64G bash