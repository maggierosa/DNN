import numpy as np
import math
import itertools
import glob
import pandas
from sklearn.model_selection import train_test_split

def factors(n):
    "Returns the pairs of all factors of a number (n) as well as their distance, respectively"
    results = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            results.append((i, int(n/i), abs(int(n/i)-i)))
    return results


def image_cut(n, size_diff=10):
    "Returns the 2D dimensions with a difference less than (size_diff) that an array of size (n) can be cut into"
    "with the least amount of loss from the array"
    banana=0
    while banana==0:
        facts = factors(n)
        mini = [j[2] for j in facts]
        if min(mini)<=size_diff:
            done = facts[mini.index(min(mini))]
            return done, done[0]*done[1]
        n = n-1

def load_xyz(directory, trajectories, num_atoms):
    "Loads and processes xyz files and labels from a list of trajectories (trajectories)"
    "Assumes xyz files are in /directory/trajectory/xyz/*.xyz and labels are in /directory/trajectory/labels.txt"
    for t in trajectories:
        xyzfiles = directory + t + "/xyz/*.xyz"
        labels_file = directory + t + "/labels.txt"
        tmp=np.empty((len(glob.glob(xyzfiles)),num_atoms,3))
        
        # Read all data files into data array
        tmp_labels = np.loadtxt(labels_file)
        i=0
        for file in sorted(glob.glob(xyzfiles)):
            tmp[i,:,:]=np.loadtxt(file)
            i+=1
        if t==trajectories[0]:
            traj = tmp
            labels = tmp_labels
        else:
            traj = np.concatenate((traj, tmp), axis=0)
            labels = np.concatenate((labels, tmp_labels))
    return traj, labels

def load_zmat(directory, trajectories, num_atoms):
    "Loads and processes zzz files and labels from a list of trajectories (trajectories)"
    "Assumes zzz files are in /directory/trajectory/zzz/*.zzz and labels are in /directory/trajectory/labels.txt"
    for t in trajectories:
        zzzfiles = directory + t + "/zzz/*.zzz"
        labels_file = directory + t + "/labels.txt"
        tmp=np.zeros((len(glob.glob(zzzfiles)),num_atoms,3))
        
        # Read all data files into data array
        tmp_labels = np.loadtxt(labels_file)
        i=0
        for file in sorted(glob.glob(zzzfiles)):
            df = pandas.read_table(file, 
                       delim_whitespace=True, 
                       header=None, 
                       usecols=range(3), 
                       engine='python')
            df[df.isnull().any(axis=1)]=0
            a = np.asarray(df)
            a[np.where(~a.any(axis=1))[0]]=a[1]
            tmp[i,:,:]=a
            i+=1
        if t==trajectories[0]:
            traj = tmp
            labels = tmp_labels
        else:
            traj = np.concatenate((traj, tmp), axis=0)
            labels = np.concatenate((labels, tmp_labels))
    return traj, labels

def traj_to_img( traj, dim_diff, image=False):
    "Converts the trajectory into an image with dimensions differing less than (dim_diff)"
    "setting image to True will squeeze the pixels into 0 to 255 to form a true image"
    "CAUTION: This will possibly cut a piece of the molecule from the end"

    img_rows = image_cut(traj.shape[1], size_diff=dim_diff)[0][0]
    img_cols = image_cut(traj.shape[1], size_diff=dim_diff)[0][1]
    nb_atoms = image_cut(traj.shape[1], size_diff=dim_diff)[1]

    traj = traj[:,:nb_atoms,:]

    if image:
        extrema=np.zeros((2,traj.shape[1],3))
        for i in range(3):
            extrema[0,i]=traj[:,i].min()
            extrema[1,i]=traj[:,i].max()
    
        for y in range(traj.shape[0]):
            for i in range(3):
                traj[y, i] = (255.0*(traj[y, i] - extrema[0,i])/(extrema[1,i]-extrema[0,i]))

    traj = np.reshape(np.transpose(traj, (1,0,2)), (img_rows,img_cols,traj.shape[0],traj.shape[2]))
    traj = np.transpose(traj, (2,0,1,3))
    return traj
