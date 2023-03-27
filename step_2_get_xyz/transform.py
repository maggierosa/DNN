import MDAnalysis
import numpy as np
import sys
sys.path.append('..')
from trajectory_tools.tools_small import *
import h5py

dim_diff = 10
psf1 = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/occluded/just_protein.psf'
traj1_scrambled = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/occluded/traj_scrambled.dcd'
out_labels_class1 = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/occluded/labels.dat'
out_data_class1 = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/occluded/coordinates.dat'

psf2 = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/outward/just_protein.psf'
traj2_scrambled = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/outward/traj_scrambled.dcd'
out_labels_class2 = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/outward/labels.dat'
out_data_class2 = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/outward/coordinates.dat'

print('reading trajectories')
 
u_class1 = MDAnalysis.Universe(psf1,traj1_scrambled)

u_class2 = MDAnalysis.Universe(psf2, traj2_scrambled)

print('extracting coordinates')

class1_XYZ = []
protein = u_class1.select_atoms("all")
for ts in u_class1.trajectory:
    class1_XYZ.append(protein.positions)
class1_XYZ = np.array(class1_XYZ)

class2_XYZ = []
protein = u_class2.select_atoms("all")
for ts in u_class2.trajectory:
    class2_XYZ.append(protein.positions)
class2_XYZ = np.array(class2_XYZ)

class1_labels = np.zeros(class1_XYZ.shape[0])
class2_labels = np.ones(class2_XYZ.shape[0])

print('Saving')
h5f1 = h5py.File(out_data_class1, 'w')
h5f1.create_dataset('dataset_class1', data=class1_XYZ)
h5f1.close()

h5f2 = h5py.File(out_data_class2, 'w')
h5f2.create_dataset('dataset_class2', data=class2_XYZ)
h5f2.close()

h5f3 = h5py.File(out_labels_class1, 'w')
h5f3.create_dataset('labels_class1', data=class1_labels)
h5f3.close()

h5f4 = h5py.File(out_labels_class2, 'w')
h5f4.create_dataset('labels_class2', data=class2_labels)
h5f4.close()
