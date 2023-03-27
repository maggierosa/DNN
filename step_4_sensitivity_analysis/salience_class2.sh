#!/bin/bash -l
#SBATCH -p edison
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

conda activate /home/agp2004/anaconda3/envs/temp_env

export CUDA_VISIBLE_DEVICES='0'

block=20
class_index=1

for i in $(seq 0 $block 780)
do
    echo $i
    python3 salience_class2.py $i $block $class_index
done
