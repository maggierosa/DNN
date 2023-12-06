##### Step 0: Import Libraries 
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
sys.path.append('/home/hex4001/DNN/')
from trajectory_tools.tools import *
from nets.densenet161 import *   ##for atom selection with a larger size >256
#from nets.densenet161_CA import *   ##for atom selection with a smaller size <256

from vis.utils import utils
from vis.visualization import *

import os


## User parameters ####################
##    Also check the user parameters in the 
##    Learning_rate decay and model building sections
lr = float(sys.argv[1])
print('Initial learning rate is %f'%lr)
nb_epoch = 30 # training epoch number
dim_diff = 10 # dimension difference, when you convert list to image 1D to 2D. Difference between x and y is 10(no need to change)
num_classes = 3 # Number of classes (User input)
stride=1 # stride of sample input

checkpoint_load = './checkpoints_stride10_lr0.0001/' # path to the weight in previous training section. Not used in a newly built DNN
checkpoint_save = './checkpoints_fullset_lr%s/'%(stride,lr) # path to weight
Data_DIR = '/athena/hwlab/scratch/mar4026/DNN/step_2_get_xyz/'

if checkpoint_load[-1]!='/': checkpoint_load=checkpoint_load+'/'
if checkpoint_save[-1]!='/': checkpoint_save=checkpoint_save+'/'
if Data_DIR[-1]!='/': Data_DIR=Data_DIR+'/'

# Input  labels
class1_labels_file = Data_DIR + 'labels_OFS.dat'
class2_labels_file = Data_DIR + 'labels_OcS.dat'
class3_labels_file = Data_DIR + 'labels_IFS.dat'

# Input data -- without data augumentation 
class1_data_files = [ Data_DIR + 'coordinates_OFS.dat' ]
class2_data_files = [ Data_DIR + 'coordinates_OcS.dat' ]
class3_data_files = [ Data_DIR + 'coordinates_IFS.dat' ]

'''
# Input data -- with data augumentation 
class1_data_files = [ Data_DIR+'coordinates_OFS1.dat', Data_DIR+'coordinates_OFS2.dat', Data_DIR+'coordinates_OFS3.dat' ]
class2_data_files = [ Data_DIR+'coordinates_OcS1.dat', Data_DIR+'coordinates_OcS2.dat', Data_DIR+'coordinates_OcS3.dat' ]
class3_data_files = [ Data_DIR+'coordinates_IFS1.dat', Data_DIR+'coordinates_IFS2.dat', Data_DIR+'coordinates_IFS3.dat' ]
## To do data augumentation, multiple data files are loaded, where the same 
## frame in different files belong to the same protein structure but different scrambles
'''

if type(class1_data_files)!=list: class1_data_files=[class1_data_files]
if type(class2_data_files)!=list: class2_data_files=[class2_data_files]
if type(class3_data_files)!=list: class3_data_files=[class3_data_files]
######################################

if not os.path.exists(checkpoint_save) : os.makedirs(checkpoint_save)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
K.tensorflow_backend._get_available_gpus()

# Import labels
h5f = h5py.File(class1_labels_file,'r')
CLASS1_label = h5f['labels'][::1]+0
h5f.close()

h5f = h5py.File(class2_labels_file,'r')
CLASS2_label = h5f['labels'][::1]+0
h5f.close()

h5f = h5py.File(class3_labels_file,'r')
CLASS3_label = h5f['labels'][::1]+0
h5f.close()

# Obtain total number of sample (total_id)
labels = np.concatenate((CLASS1_label, CLASS2_label, CLASS3_label))
total_id =  np.arange(len(labels))

# Split into training, validation and test set 
# split of test from train 
X_train_id, X_test_id, Y_train, Y_test = train_test_split(total_id, labels, test_size=0.20, random_state=42) 
# random_state makes sure you keep the same samples in each so that next time not in valid
# split of valid from train 
X_train_id, X_valid_id, Y_train, Y_valid = train_test_split(X_train_id, Y_train, test_size=0.30, random_state=66)

# Convert each element in Y to vector that has length of # of classes 
# i.e. Y used to be 0,1,2, when y=0 converts to vector [1,0,0], y=1 converts to vector [0,1,0], when y=2 converts to vector [0,0,1]
Y_train_temp = np_utils.to_categorical(Y_train, num_classes)[::stride,:]
Y_valid_temp = np_utils.to_categorical(Y_valid, num_classes)[::stride,:]
Y_test_temp = np_utils.to_categorical(Y_test, num_classes)[::stride,:]

# Import data:
## Apply the same subset assignment to each copy of the dataset
## so that the different scrambles that belongs to the same protein
## will always be assigned to the same subset
dataset_imported = 0
data_augument_fold = len(class1_data_files)
for i in range(data_augument_fold):
    h5f = h5py.File(class1_data_files[i],'r')
    CLASS1_DATA = h5f['dataset'][::1]+0
    h5f.close()
    h5f = h5py.File(class2_data_files[i],'r')
    CLASS2_DATA = h5f['dataset'][::1]+0
    h5f.close()
    h5f = h5py.File(class3_data_files[i],'r')
    CLASS3_DATA = h5f['dataset'][::1]+0
    h5f.close()
    data_set_temp = np.concatenate((CLASS1_DATA, CLASS2_DATA, CLASS3_DATA), axis=0)
    del CLASS1_DATA, CLASS2_DATA, CLASS3_DATA #clean memory
    if dataset_imported == 0:
        dataset_shape = np.array(data_set_temp.shape)
        dataset_shape[0] = 0
        total_train = np.zeros(dataset_shape)
        total_valid = np.zeros(dataset_shape)
        total_test = np.zeros(dataset_shape)
        dataset_imported = 1
    data_train = data_set_temp[X_train_id,:,:][::stride,:,:]
    data_valid = data_set_temp[X_valid_id,:,:][::stride,:,:]
    data_test = data_set_temp[X_test_id,:,:][::stride,:,:]
    del data_set_temp #clean memory
    total_train = np.concatenate((total_train,data_train), axis=0)
    total_valid = np.concatenate((total_valid,data_valid), axis=0)
    total_test = np.concatenate((total_test,data_test), axis=0)
    del data_train, data_valid, data_test #clean memory

# make replicas of Y
Y_train = Y_train_temp
Y_valid = Y_valid_temp
Y_test = Y_test_temp
for i in range(data_augument_fold-1):
    Y_train = np.concatenate((Y_train,Y_train_temp), axis=0)
    Y_valid = np.concatenate((Y_valid,Y_valid_temp), axis=0)
    Y_test = np.concatenate((Y_test,Y_test_temp), axis=0)

# convert to visual representation
X_train = traj_to_img(total_train, dim_diff=dim_diff)
X_valid = traj_to_img(total_valid, dim_diff=dim_diff)
X_test = traj_to_img(total_test, dim_diff=dim_diff)

#check data shape
print("X_train:",X_train.shape)
print("X_valid:",X_valid.shape)
print("X_test:",X_test.shape)
print("Y_train:",Y_train.shape)
print("Y_valid:",Y_valid.shape)
print("Y_test:",Y_test.shape)

#clean up
del Y_train_temp, Y_valid_temp, Y_test_temp
del total_train, total_valid, total_test

# data normalization
train_datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
train_datagen.fit(X_train)
norm_parameters = np.concatenate((train_datagen.mean,train_datagen.std)).reshape([2,-1])
np.savetxt(checkpoint_save + "normalization.dat", norm_parameters.astype(str),fmt='%s')

X_train = train_datagen.standardize(X_train)
X_valid = train_datagen.standardize(X_valid)
X_test = train_datagen.standardize(X_test)

# Learning_rate decay
def lr_schedule(epoch, lr):
    if epoch >= 10: # the starting epoch for Learning_rate decay
        lr = lr * np.exp(-0.1) # the rate of learning_rate decay per epoch
    print("the learning rate is set to %f" %lr)
    return lr
lrsc = LearningRateScheduler(lr_schedule)

# Report the weights: 1. the newest, 2. the best performance on the validation set
checkpointer_bvl = ModelCheckpoint(filepath=checkpoint_save + 'weights_best_val_loss.hdf5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
checkpointer_new = ModelCheckpoint(filepath=checkpoint_save + 'weights_newest.hdf5', verbose=1, save_weights_only=True, save_best_only=False)

# Build the model
batch_size = 16 
def buildmodel(X_train, Y_train, X_valid, Y_valid, num_classes=num_classes):
    if __name__ == '__main__':
        img_rows, img_cols =  X_train.shape[1], X_train.shape[2] # Resolution of inputs
        channel = 3
        model = densenet161_model(img_rows=img_rows, 
            img_cols=img_cols, color_type=channel, 
            num_classes=num_classes, nb_dense_block=3, 
            growth_rate=48, nb_filter=96, reduction=0.5, 
            dropout_rate=0.0, weight_decay=0.0,learn_rate=lr)
        return model

model = buildmodel(X_train, Y_train, X_valid, Y_valid)
model.load_weights(checkpoint_load + 'weights_newest.hdf5') #load the newest weight from last step with stride10 to continue training
print(model.summary())

# Fit the model  
history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=nb_epoch, batch_size=batch_size, callbacks=[checkpointer_bvl,checkpointer_new,lrsc],shuffle=True)

# save the history of accuracy and loss 
hist_df = pd.DataFrame(history.history)
hist_csv_file = checkpoint_save + 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# check the performance of DNN on training set 
train_result = model.predict(X_train, verbose=0)
accuracy = 100*np.sum(np.argmax(train_result, axis=1)==np.argmax(Y_train, axis=1))/Y_train.shape[0]
print("Overall train set accuracy is %0.4f%%"%accuracy)

class1_ind = np.where(np.argmax(Y_train, axis=1)==0)[0]
class1_train_result = train_result[class1_ind,:]
accuracy = 100*np.sum(np.argmax(class1_train_result, axis=1)==np.argmax(Y_train[class1_ind], axis=1))/Y_train[class1_ind].shape[0]
print('class1 train set size is ',len(class1_ind))
print("class1 train set accuracy is %0.4f%%"%accuracy)

class2_ind = np.where(np.argmax(Y_train, axis=1)==1)[0]
class2_train_result = train_result[class2_ind,:]
accuracy = 100*np.sum(np.argmax(class2_train_result, axis=1)==np.argmax(Y_train[class2_ind], axis=1))/Y_train[class2_ind].shape[0]
print('class2 train set size is ',len(class2_ind))
print("class2 train set accuracy is %0.4f%%"%accuracy)

class3_ind = np.where(np.argmax(Y_train, axis=1)==2)[0]
class3_train_result = train_result[class3_ind,:]
accuracy = 100*np.sum(np.argmax(class3_train_result, axis=1)==np.argmax(Y_train[class3_ind], axis=1))/Y_train[class3_ind].shape[0]
print('class3 train set size is ',len(class3_ind))
print("class3 train set accuracy is %0.4f%%"%accuracy)

#valid set
valid_result = model.predict(X_valid, verbose=0)
accuracy = 100*np.sum(np.argmax(valid_result, axis=1)==np.argmax(Y_valid, axis=1))/Y_valid.shape[0]
print("Overall valid set accuracy is %0.4f%%"%accuracy)

class1_ind = np.where(np.argmax(Y_valid, axis=1)==0)[0]
class1_valid_result = valid_result[class1_ind,:]
accuracy = 100*np.sum(np.argmax(class1_valid_result, axis=1)==np.argmax(Y_valid[class1_ind], axis=1))/Y_valid[class1_ind].shape[0]
print('class1 valid set size is ',len(class1_ind))
print("class1 valid set accuracy is %0.4f%%"%accuracy)

class2_ind = np.where(np.argmax(Y_valid, axis=1)==1)[0]
class2_valid_result = valid_result[class2_ind,:]
accuracy = 100*np.sum(np.argmax(class2_valid_result, axis=1)==np.argmax(Y_valid[class2_ind], axis=1))/Y_valid[class2_ind].shape[0]
print('class2 valid set size is ',len(class2_ind))
print("class2 valid set accuracy is %0.4f%%"%accuracy)

class3_ind = np.where(np.argmax(Y_valid, axis=1)==2)[0]
class3_valid_result = valid_result[class3_ind,:]
accuracy = 100*np.sum(np.argmax(class3_valid_result, axis=1)==np.argmax(Y_valid[class3_ind], axis=1))/Y_valid[class3_ind].shape[0]
print('class3 valid set size is ',len(class3_ind))
print("class3 valid set accuracy is %0.4f%%"%accuracy)


#test set
test_result = model.predict(X_test, verbose=0)
accuracy = 100*np.sum(np.argmax(test_result, axis=1)==np.argmax(Y_test, axis=1))/Y_test.shape[0]
print("Overall test set accuracy is %0.4f%%"%accuracy)

class1_ind = np.where(np.argmax(Y_test, axis=1)==0)[0]
class1_test_result = test_result[class1_ind,:]
accuracy = 100*np.sum(np.argmax(class1_test_result, axis=1)==np.argmax(Y_test[class1_ind], axis=1))/Y_test[class1_ind].shape[0]
print('class1 test set size is ',len(class1_ind))
print("class1 test set accuracy is %0.4f%%"%accuracy)

class2_ind = np.where(np.argmax(Y_test, axis=1)==1)[0]
class2_test_result = test_result[class2_ind,:]
accuracy = 100*np.sum(np.argmax(class2_test_result, axis=1)==np.argmax(Y_test[class2_ind], axis=1))/Y_test[class2_ind].shape[0]
print('class2 test set size is ',len(class2_ind))
print("class2 test set accuracy is %0.4f%%"%accuracy)

class3_ind = np.where(np.argmax(Y_test, axis=1)==2)[0]
class3_test_result = test_result[class3_ind,:]
accuracy = 100*np.sum(np.argmax(class3_test_result, axis=1)==np.argmax(Y_test[class3_ind], axis=1))/Y_test[class3_ind].shape[0]
print('class3 test set size is ',len(class3_ind))
print("class3 test set accuracy is %0.4f%%"%accuracy)
