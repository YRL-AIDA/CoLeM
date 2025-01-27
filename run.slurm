#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=79G
#SBATCH --time=14-07:30:00
#SBATCH --job-name=colem

source venv/bin/activate &&\
    pip install -r requirements.txt &&\
    python3 train.py 2> logs/error_train.log
