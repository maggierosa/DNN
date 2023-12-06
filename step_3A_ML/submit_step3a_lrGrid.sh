#!/bin/bash -l
#SBATCH -p hwlab-rocky-gpu,cryo-gpu-v100-low,scu-gpu,cryo-gpu-low,cryo-gpu-p100-low
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

conda activate ambrose_neural_nets

## You'll try a series of learning rates, and choose the one that gives the best performance.
## In this step, the DNN is newly built. 
## The newly built DNN takes only a small portion of the input samples rather than the full dataset, so it's easier to learn something to start with.

#lr='0.00005'
lr='0.0001'
#lr='0.0002'
#lr='0.0005'
#lr='0.001'
#lr='0.002'

python ml_new_train.py $lr > trainlog_lrgrid_lr${lr}.out

