#!/bin/bash -l
#SBATCH -p hwlab-rocky-gpu,cryo-gpu-v100-low,scu-gpu,cryo-gpu-low,cryo-gpu-p100-low
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

conda activate ambrose_neural_nets

## In this step, load the DNN with the best performance on full dataset 
## and keep training with regularizations to fight with overfitting

lr='0.00005'
python ml_load_train_regularize.py $lr > trainlog_regularize_lr${lr}.out


