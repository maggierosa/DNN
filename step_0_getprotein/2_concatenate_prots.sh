#!/bin/bash -l
#SBATCH -p hwlab-rocky-gpu,cryo-gpu-v100-low,scu-gpu,cryo-gpu-low,cryo-gpu-p100-low
#SBATCH -n1
#SBATCH --mem=100G

module load vmd/vmd_1.9.3_TEXT_CPU_ONLY
catdcd -o combined.dcd trajs/*dcd

exit
