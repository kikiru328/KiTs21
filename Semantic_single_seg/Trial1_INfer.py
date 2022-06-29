# module import
print("###### Module import #####")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from glob import glob
from tqdm import tqdm
import shutil
import random as r
import math

import nibabel as nib
import nilearn
import nilearn.plotting as nlplt
from nilearn import plotting
import warnings
warnings.filterwarnings('ignore')

import cv2 as cv
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps  
import skimage.io as io
import skimage.color as color


from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input ,BatchNormalization , Activation 
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
# from keras.layers.merge import concatenate
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
# from keras import optimizers 
from tensorflow.keras import optimizers
print("###### DONE #####")

# Data Preprocessing
print("###### DATA_PREPROCESSING ######")
path = '/root/project/kits/kits21/kits21/data'
p = sorted(os.listdir(path))
# p.remove('.DS_Store')
# p.remove('kits.json')
def Data_Preprocessing(modalities_dir):
    all_modalities = []    
    for modality in modalities_dir:      
        nifti_file   = nib.load(modality)
        kidney_numpy  = np.asarray(nifti_file.dataobj)    
        all_modalities.append(kidney_numpy)
    kidney_affine   = nifti_file.affine
    all_modalities = np.array(all_modalities)
    all_modalities = np.rint(all_modalities).astype(np.int16)
    all_modalities = all_modalities[:, :, :, :]
    # all_modalities = np.transpose(all_modalities)
    all_modalities = all_modalities.transpose((1,2,3,0))
    return all_modalities

print("##### INPUT_DATA #####")
Input_Data = []
for i in tqdm(p[100:106]):
    kidney_dir = f"{path}/{i}/"
    img_ = glob(f'{kidney_dir}imaging.nii.gz')
    and_ = glob(f'{kidney_dir}aggregated_AND_seg.nii.gz')
    modalities_dir = [img_[0], and_[0]]
    p_data = Data_Preprocessing(modalities_dir=modalities_dir)
    Input_Data.append(p_data)
    
print('len > ', len(Input_Data), 'shape > ',Input_Data[0].shape)


def Data_Concatenate(Input_Data):
    counter=0
    Output= []
    for i in tqdm(range(2)):
        # print('$')
        c=0
        counter=0
        for ii in range(len(Input_Data)):
            if (counter != len(Input_Data)):
                a= Input_Data[counter][:,:,:,i]
                # print('a={}'.format(a.shape))
                b= Input_Data[counter+1][:,:,:,i]
                # print('b={}'.format(b.shape))
                if(counter==0):
                    c= np.concatenate((a, b), axis=0)
                    # print('c1={}'.format(c.shape))
                    counter= counter+2
                else:
                    c1= np.concatenate((a, b), axis=0)
                    c= np.concatenate((c, c1), axis=0)
                    # print('c2={}'.format(c.shape))
                    counter= counter+2
        c= c[:,:,:,np.newaxis]
        Output.append(c)
    return Output

InData= Data_Concatenate(Input_Data)


AIO= concatenate(InData, axis=3)
AIO=np.array(AIO,dtype='float32')
TR=np.array(AIO[:,:,:,0],dtype='float32')
TRL=np.array(AIO[:,:,:,1],dtype='float32')

from sklearn.model_selection import train_test_split
X_train , X_test, Y_train, Y_test = train_test_split(TR, TRL, test_size=0.15, random_state=32)
AIO=TRL=0

print("###### DONE ######")

# MODEL
print("###### MODELS ######")
def Convolution(input_tensor,filters):
    
    x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    return x

def model(input_shape):
    
    inputs = Input((input_shape))
    
    conv_1 = Convolution(inputs,32)
    maxp_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_1)
    
    conv_2 = Convolution(maxp_1,64)
    maxp_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_2)
    
    conv_3 = Convolution(maxp_2,128)
    maxp_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_3)
    
    conv_4 = Convolution(maxp_3,256)
    maxp_4 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_4)
    
    conv_5 = Convolution(maxp_4,512)
    upsample_6 = UpSampling2D((2, 2)) (conv_5)
    
    conv_6 = Convolution(upsample_6,256)
    upsample_7 = UpSampling2D((2, 2)) (conv_6)
    
    upsample_7 = concatenate([upsample_7, conv_3])
    
    conv_7 = Convolution(upsample_7,128)
    upsample_8 = UpSampling2D((2, 2)) (conv_7)
    
    conv_8 = Convolution(upsample_8,64)
    upsample_9 = UpSampling2D((2, 2)) (conv_8)
    
    upsample_9 = concatenate([upsample_9, conv_1])
    
    conv_9 = Convolution(upsample_9,32)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv_9)
    
    model = Model(inputs=[inputs], outputs=[outputs]) 
    
    return model

# Loding the Light weighted CNN
model = model(input_shape = (512,512,1))
model.summary()


# Computing Dice_Coefficient
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

model.load_weights('code/kits/model/Trial_1_AND.h5')


pref_Tumor = model.predict(TR)

print('################## WORKS! ########################')


import pickle
with open('code/kits/model/Tria1_INFER.pkl','wb') as F:
    pickle.dump(pref_Tumor,F)






# Compiling the model 
# Adam=optimizers.Adam(lr=0.001)
# model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy',dice_coef,precision,sensitivity,specificity])


# print('\n')
# print('######### FIT ###########')
# print('BATCH_SIZE = 32')
# print('EPOCHS = 40')
# # Fitting the model over the data
# history = model.fit(X_train,Y_train,batch_size=32,epochs=40,validation_split=0.20,verbose=1,initial_epoch=0)

# # EVALUATIOn
# print('###### EVAL ######')
# # Evaluating the model on the training and testing data 
# print('### TRAIN ###')
# model.evaluate(x=X_train, y=Y_train, batch_size=32 , verbose=1, sample_weight=None, steps=None)
# print('### TEST ###')
# model.evaluate(x=X_test, y=Y_test, batch_size=32, verbose=1, sample_weight=None, steps=None)
# model.save('/root/project/code/kits/model/Trial_1_AND.h5')
# print('###### MODEL SAVED ######')

# hist_df = pd.DataFrame(history.history) 
# hist_csv_file = '/root/project/code/kits/model/Trial_1_AND.csv'
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
# print('###### history SAVED ######')dw