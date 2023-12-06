#!/bin/bash 

## This script goes through trajectories of simulations to obtain the trajectories of protein-only and align them with a template
## Things that need to be modified and personalized: 
##   1. "CVscriptPath": the path to the folder where you keep these scripts
##   2. trajectory ids
##  *3. inside the file "$CVscriptPath/submit_getprot.sh"

CVscriptPath='/home/hex4001/DNN/step_0_getprotein' # the path to the folder where you keep these scripts
for i in {1..11}  #trajectory ids
do
   cp $CVscriptPath/submit_getprot.sh ./getprotsingle_temp${i}.sh
   sed -i "s/trajidinput/$i/" ./getprotsingle_temp${i}.sh
   sbatch ./getprotsingle_temp${i}.sh
   rm ./getprotsingle_temp${i}.sh
done
