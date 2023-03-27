#!/bin/bash -l
#SBATCH -p panda,edison,hwlab_reserve
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G

echo 'Goodbye'
conda activate /home/agp2004/anaconda3/envs/temp_env
echo 'Hello'
python3 transform.py 
echo 'It Worked'
