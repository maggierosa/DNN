import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
import sys

psf = str(sys.argv[1])
dcd = str(sys.argv[2])
out_dcd = str(sys.argv[3])

mobile = mda.Universe(psf, dcd)
ref = mda.Universe(psf, dcd)

mobile.trajectory[-1]  # set mobile trajectory to last frame
ref.trajectory[0]  # set reference trajectory to first frame

# Do the first alignment with no weights
aligner = align.AlignTraj(mobile, ref, select='name CA', in_memory=True).run()

# Extract C-alpha coordinates
# ---------
temp=[]
protein = mobile.select_atoms("name CA")
for ts in mobile.trajectory:
    temp.append(protein.positions)

temp = np.asarray(temp)
# ---------

# get all c-alpha distances
# ---------
from itertools import combinations
Calphas = np.array(list(range(temp.shape[1])))
pairs = list(combinations(Calphas,2))

dist_mat = np.array([np.linalg.norm((temp[:,pair[0],:]-temp[:,pair[1],:]), axis=1) for pair in pairs])
pairs = np.asarray(pairs)
# ---------

# Calculate mean standard deviation for each residue
# ---------
stds=[]
for i in Calphas:
    stds.append(np.std(dist_mat[np.where((pairs[:,1]==i) | (pairs[:,0]==i))[0]], axis=1).mean())
stds = np.asarray(stds)
# ---------

mobile = mda.Universe(psf, dcd)
ref = mda.Universe(psf, dcd)

mobile.trajectory[-1]  # set mobile trajectory to last frame
ref.trajectory[3000]  # set reference trajectory to first frame

# Align using the inverse of the standard deviation as weights
aligner = align.AlignTraj(mobile, ref, select='name CA', weights=(1/stds**2), filename=out_dcd).run() # increase power to make sharper, decrease to make flatter
