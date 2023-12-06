#!/bin/bash -l 
#SBATCH -p hwlab-rocky-gpu,cryo-gpu-v100-low,scu-gpu,cryo-gpu-low,cryo-gpu-p100-low
#SBATCH --mem=100G

## This script applies position and orientation scrambling on the protein trajectory
##   For data augumentation, you may obtain multiple scrambling from the same set of protein.
##   For the valification test of saliency obtained in step_4, you may rebuilt the network by keeping only a partial of the protein in this step.
## Things that need to be modified and personalized: 
##   1. "CVscriptPath": the path to the folder where you keep these scripts
##   2. input psf file
##   3. output folder path
##   4. input trajatory file
##  *5. inside the file "$CVscriptPath/extract_and_scramble_protein_notsalient.tcl"
##   6. The name of output scrambled trajctory(s)


module load vmd/vmd_1.9.3_TEXT_CPU_ONLY

CVscriptPath='/home/hex4001/DNN/step_1_scramble' # the path to the folder where you keep these scripts
psfname='/home/hex4001/DNN/protein_structures/chol45/chol_StarD4_prot_only.psf' # input psf file
outputpath='/home/hex4001/DNN/protein_onlysalient/chol45/' # output folder path
xtcname='/home/hex4001/DNN/protein_structures/chol45/combined.xtc' # input trajatory file
vmd -dispdev text -e $CVscriptPath/extract_and_scramble_protein_notsalient.tcl -args $psfname $xtcname $outputpath/scramble_os.psf $outputpath/scramble1_os.dcd
vmd -dispdev text -e $CVscriptPath/extract_and_scramble_protein_notsalient.tcl -args $psfname $xtcname $outputpath/scramble_os.psf $outputpath/scramble2_os.dcd
vmd -dispdev text -e $CVscriptPath/extract_and_scramble_protein_notsalient.tcl -args $psfname $xtcname $outputpath/scramble_os.psf $outputpath/scramble3_os.dcd


exit
