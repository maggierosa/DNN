#!/bin/bash -l
#SBATCH -p hwlab-rocky-gpu
#SBATCH --mem=500G
#SBATCH --gres=gpu:2

conda activate ambrose_neural_nets

export CUDA_VISIBLE_DEVICES='0'

block=20
class_index=0
# class index of interest, the script is to answer the question
# "what is the features that help DNN successfully classify these samples as *class X* ?" 
# This class index usually matches the real index of the input samples in this step

vis_file='/athena/hwlab/scratch/mar4026/DNN/step_3B_visualize/MFSD2A_IFS_vis_val_FJ_remove.dat'
# vis file that is empty

data_file='/athena/hwlab/scratch/mar4026/DNN/step_3B_visualize/MFSD2A_IFS_test_score_val_FJ_remove.dat'
# data file with best 1000 samples

model_file='/athena/hwlab/scratch/mar4026/DNN/step_3B_visualize/model_1_MFSD2A_best_loss_val_0.6.h5'
# model obtained in step 3B, with linear output layer

for i in $(seq 0 $block 1000)
do
    echo $i
    python3 visualize_salience.py $i $block $class_index $vis_file $data_file $model_file
done
