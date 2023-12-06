#!/bin/bash -l
#SBATCH -p hwlab-rocky-gpu,cryo-gpu-v100-low,scu-gpu,cryo-gpu-low,cryo-gpu-p100-low
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

conda activate ambrose_neural_nets

## In this step, load the DNN with the best performance on 1/100 sample
## and keep training with 1/10 sample

lr='0.0001'
python ml_load_train_stride10.py $lr > trainlog_stride10_lr${lr}.out


