#!/bin/bash -l
#SBATCH -p hwlab-rocky-gpu,cryo-gpu-v100-low,scu-gpu,cryo-gpu-low,cryo-gpu-p100-low
#SBATCH -n1
#SBATCH --mem=100G

## This script goes through a trajectory of simulation to obtain the protein and align them with a template
## Things that need to be modified and personalized: 
##   1. "CVscriptPath": the path to the folder where you keep these scripts
##   2. How many frames to be discarded at the beginning of each trajectory
##   3. The "input" path to the folder with the trajectory of simulation
##   4. The "output" path to the folder that stores the protein trajectories ready for DNN analysis
##   5. A template protein to align the trajectory with
##  *6. Inside the file "$CVscriptPath/getprot_wrap.tcl"

CVscriptPath='/home/hex4001/DNN/step_0_getprotein'

i=trajidinput
trajid=`printf %04d $i`

firstframe=0 # How many frames to be discarded at the beginning of each trajectory
fromdir="./traj_stride2/traj${trajid}" # The "input" path to the folder with the trajectory of simulation
aimdir="./prot_stride2/traj${trajid}" # The "output" path to the folder that stores the protein trajectories ready for DNN analysis
mkdir -p $aimdir
template='/athena/hwlab/scratch/hex4001/StarD4/f.pdb' # A template protein to align the trajectory with

psfname=`ls ${fromdir}/*psf`
dcdname=`ls ${fromdir}/*[dx][ct][dc]`

module load vmd/vmd_1.9.3_TEXT_CPU_ONLY
vmd3 -dispdev text -e $CVscriptPath/getprot_wrap.tcl -args $psfname $dcdname $firstframe $template ${aimdir}/traj${trajid}

exit
