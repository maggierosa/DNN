#!/bin/bash -l
#SBATCH -p hwlab-rocky-gpu,cryo-gpu-v100-low,scu-gpu,cryo-gpu-low,cryo-gpu-p100-low
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G

## This script turns trajectory into a matrix of xyz-t of Calpha only
## Inputs:
##   scramble_os.psf
##   scrambled1.dcd
##   grouplabel ---- the class type id (0,1,2...)
## Outputs:
##   labels.dat ---- class-t vector
##   coordinates1.dat ---- xyz-t matrix

conda activate ambrose_neural_nets
ScriptDir='/home/hex4001/DNN/step_2_get_xyz/'
grouplabel=1
python3 ${ScriptDir}/transform_CA.py scrambled.psf scrambled1.dcd labels.dat CA_coordinates1.dat $grouplabel
python3 ${ScriptDir}/transform_CA.py scrambled.psf scrambled2.dcd labels.dat CA_coordinates2.dat $grouplabel
python3 ${ScriptDir}/transform_CA.py scrambled.psf scrambled3.dcd labels.dat CA_coordinates3.dat $grouplabel

exit
