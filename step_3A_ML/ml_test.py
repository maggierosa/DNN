import numpy as np
import h5py
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers import *
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.models import load_model

from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import sys
from trajectory_tools.tools import *
from nets.densenet161_CYC import *

from vis.utils import utils
from vis.visualization import *

import os


## user parameters ####################
class1_labels_file = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/occluded/labels.dat'
class1_data_file = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/occluded/coordinates.dat'
class2_labels_file = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/outward/labels.dat'
class2_data_file = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/outward/coordinates.dat'
checkpoint_stem = '/athena/hwlab/scratch/agp2004/MFSD2A/DNN/data/'

dim_diff = 10
num_classes = 2
stride=1
######################################

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
K.tensorflow_backend._get_available_gpus()

h5f1 = h5py.File(class1_data_file,'r')
CLASS1_train_set = h5f1['dataset_class1'][::stride]
h5f1.close()

h5f2 = h5py.File(class1_labels_file,'r')
class1_train_set = h5f2['labels_class1'][::stride]
h5f2.close()

h5f3 = h5py.File(class2_data_file,'r')
CLASS2_train_set = h5f3['dataset_class2'][::stride]
h5f3.close()

h5f4 = h5py.File(class2_labels_file,'r')
class2_train_set = h5f4['labels_class2'][::stride]
h5f4.close()

traj_xyz = np.concatenate((CLASS1_train_set, CLASS2_train_set), axis=0)
labels = np.concatenate((class1_train_set, class2_train_set))

traj = traj_xyz

traj = traj_to_img(traj, dim_diff=dim_diff)
X_train, X_test, Y_train, Y_test = train_test_split(traj, labels, test_size=0.20, random_state=42)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.30, random_state=66)
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_valid = np_utils.to_categorical(Y_valid, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
    )

train_datagen.fit(X_train)

X_train = train_datagen.standardize(X_train)
X_valid = train_datagen.standardize(X_valid)
X_test = train_datagen.standardize(X_test)
    

def fitmodel(X_train, Y_train, X_valid, Y_valid):
    if __name__ == '__main__':

        img_rows, img_cols =  X_train.shape[1], X_train.shape[2] # Resolution of inputs
        channel = 3
        num_classes = 2 
        batch_size = 6
        nb_epoch = 100
        lr=1e-4

        # Load our model
        model = densenet161_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.5, dropout_rate=0.2, weight_decay=1e-4)

        def lr_schedule(epoch, lr):
            return lr * (0.1 ** int(epoch / 10))
    #     LearningRateScheduler(lr_schedule)

        checkpointer = ModelCheckpoint(filepath=checkpoint_stem + '_weights_stride_%s.hdf5' %stride, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
#         # Start Fine-tuning
        history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=nb_epoch, batch_size=batch_size, callbacks=[checkpointer])
        return model, history

model, history = fitmodel(X_train, Y_train, X_valid, Y_valid)

model.load_weights(checkpoint_stem + '_weights_stride_%s.hdf5' %stride)

hist_df = pd.DataFrame(history.history)
hist_csv_file = checkpoint_stem + 'history_n%s.csv' %stride
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

scores = model.predict(X_test, verbose=0)
accuracy = 100*np.sum(np.argmax(scores, axis=1)==np.argmax(Y_test, axis=1))/Y_test.shape[0]
print("Overall test set accuracy is    %",accuracy)

class1_ind = np.where(np.argmax(Y_test, axis=1)==0)
scores = model.predict(X_test[class1_ind], verbose=0)
accuracy = 100*np.sum(np.argmax(scores, axis=1)==np.argmax(Y_test[class1_ind], axis=1))/Y_test[class1_ind].shape[0]
print("class1 test set accuracy is    %",accuracy)

class2_ind = np.where(np.argmax(Y_test, axis=1)==1)
scores = model.predict(X_test[class2_ind], verbose=0)
accuracy = 100*np.sum(np.argmax(scores, axis=1)==np.argmax(Y_test[class2_ind], axis=1))/Y_test[class2_ind].shape[0]
print("Class2 test set accuracy is    %",accuracy)

