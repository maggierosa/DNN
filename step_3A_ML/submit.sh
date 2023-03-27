#!/bin/bash -l
#SBATCH -p edison
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=30G

conda activate /home/agp2004/anaconda3/envs/temp_env

python ml_test.py
#python evaluation.py
