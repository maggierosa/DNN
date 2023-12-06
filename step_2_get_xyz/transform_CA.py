import MDAnalysis
import numpy as np
import sys
import h5py

#BASE_DIR = '/athena/hwlab/scratch/agp2004/NEW_5HT2A/DNN/'

#psf3 = BASE_DIR + 'data/ERG/just_protein.psf'
#traj3_scrambled = BASE_DIR + 'data/ERG/traj_scrambled.dcd'
#out_labels_class3 = BASE_DIR + 'data/ERG/labels.dat'
#out_data_class3 = BASE_DIR + 'data/ERG/coordinates.dat'

psf = sys.argv[1]
traj_scrambled = sys.argv[2]
out_labels = sys.argv[3]
out_data = sys.argv[4]
class_code = int(sys.argv[5])

print('reading trajectories')
 
u_class1 = MDAnalysis.Universe(psf,traj_scrambled)

print('extracting coordinates')

class1_XYZ = []
Ca = u_class1.select_atoms("name CA and not resid 24 25 26 224 223")
for ts in u_class1.trajectory:
    class1_XYZ.append(Ca.positions)
class1_XYZ = np.array(class1_XYZ)

class1_labels = np.ones(class1_XYZ.shape[0])*class_code

print('Saving')
h5f1 = h5py.File(out_data, 'w')
h5f1.create_dataset('dataset', data=class1_XYZ)
h5f1.close()

h5f2 = h5py.File(out_labels, 'w')
h5f2.create_dataset('labels', data=class1_labels)
h5f2.close()

exit
