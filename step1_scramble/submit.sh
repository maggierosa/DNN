#!/bin/bash
#SBATCH -p panda,edison,hwlab_reserve
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G

vmd_athena='/athena/hwlab/scratch/lab_data/software/vmd/vmd-1.9.3_athena/install_bin/vmd_athena'

$vmd_athena -dispdev text -e extract_and_scramble_protein.tcl -args /athena/hwlab/scratch/khelgeo/mfs_project/outward_open_popc/ml_analysis/traj_8/trajectories/occluded/prot.psf /athena/hwlab/scratch/khelgeo/mfs_project/outward_open_popc/ml_analysis/traj_8/trajectories/occluded/trajectory_prot_0.xtc ../data/occluded/just_protein.psf ../data/occluded/traj_scrambled.dcd

$vmd_athena -dispdev text -e extract_and_scramble_protein.tcl -args /athena/hwlab/scratch/khelgeo/mfs_project/outward_open_popc/ml_analysis/traj_8/trajectories/outward/prot.psf /athena/hwlab/scratch/khelgeo/mfs_project/outward_open_popc/ml_analysis/traj_8/trajectories/outward/trajectory_prot_0.xtc ../data/outward/just_protein.psf ../data/outward/traj_scrambled.dcd
