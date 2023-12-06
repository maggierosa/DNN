#!/bin/bash -l
#SBATCH -p hwlab-rocky-gpu,cryo-gpu-v100-low,scu-gpu,cryo-gpu-low,cryo-gpu-p100-low
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

conda activate ambrose_neural_nets

python evaluate.py > evaluate.out


