
# coding: utf-8

# In[1]:


from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.layers import Input, Concatenate, Conv2DTranspose, Lambda
from keras.utils import plot_model
from keras.backend import dot
import tensorflow as tf
from tensorflow.keras.backend import squeeze,expand_dims
from keras.applications import vgg16
#from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
import keras
import cv2
import numpy as np
import os


# In[2]:


model1 = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(320,320,3))
model2 = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(160,160,3))
model3 = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(240,240,3))
model1.summary()


# In[3]:


model2.layers


# In[4]:


# def VGG(weight_path=None):
model = Sequential()
# model1 = Sequential()
# model2 = Sequential()

########################################################################################33
########################################################################################33
########################################################################################33

main_input0 = Input(shape=(160,160,3), dtype='float32', name='main_input0')
scale0_layer_1 = ZeroPadding2D((1,1))(main_input0)
scale0_layer_2 = Convolution2D(64, (3, 3), activation='relu', weights = model1.layers[1].get_weights())(scale0_layer_1)
scale0_layer_3 = ZeroPadding2D((1,1))(scale0_layer_2)
scale0_layer_4 = Convolution2D(64, (3, 3), activation='relu', weights = model1.layers[2].get_weights())(scale0_layer_3)
scale0_layer_5 = ZeroPadding2D((1,1))(scale0_layer_4)
scale0_pool_1 = MaxPooling2D((3,3), strides=(2,2))(scale0_layer_5)
################ scale0_pool_1 = scale0_layer_5 ##################

scale0_layer_6 = ZeroPadding2D((1,1))(scale0_pool_1)
scale0_layer_7 = Convolution2D(128, (3, 3), activation='relu', weights = model1.layers[4].get_weights())(scale0_layer_6)
scale0_layer_8 = ZeroPadding2D((1,1))(scale0_layer_7)
scale0_layer_9 = Convolution2D(128, (3, 3), activation='relu', weights = model1.layers[5].get_weights())(scale0_layer_8)
scale0_layer_10 = ZeroPadding2D((1,1))(scale0_layer_9)
scale0_pool_2 = MaxPooling2D((3,3), strides=(2,2))(scale0_layer_10)
################ scale0_pool_2 = scale0_layer_11 ##################

scale0_layer_12 = ZeroPadding2D((1,1))(scale0_pool_2)
scale0_layer_13 = Convolution2D(256, (3, 3), activation='relu', weights = model1.layers[7].get_weights())(scale0_layer_12)
scale0_layer_14 = ZeroPadding2D((1,1))(scale0_layer_13)
scale0_layer_15 = Convolution2D(256, (3, 3), activation='relu', weights = model1.layers[8].get_weights())(scale0_layer_14)
scale0_layer_16 = ZeroPadding2D((1,1))(scale0_layer_15)
scale0_layer_17 = Convolution2D(256, (3, 3), activation='relu', weights = model1.layers[9].get_weights())(scale0_layer_16)
scale0_layer_18 = ZeroPadding2D((1,1))(scale0_layer_17)
scale0_pool_3 = MaxPooling2D((3,3), strides=(2,2))(scale0_layer_18)
################ scale0_pool_3 = scale0_layer_19 ##################

scale0_layer_20 = ZeroPadding2D((1,1))(scale0_pool_3)
scale0_layer_21 = Convolution2D(512, (3, 3), activation='relu',weights = model1.layers[11].get_weights())(scale0_layer_20)
scale0_layer_22 = ZeroPadding2D((1,1))(scale0_layer_21)
scale0_layer_23 = Convolution2D(512, (3, 3), activation='relu', weights = model1.layers[12].get_weights())(scale0_layer_22)
scale0_layer_24 = ZeroPadding2D((1,1))(scale0_layer_23)
scale0_layer_25 = Convolution2D(512, (3, 3), activation='relu', weights = model1.layers[13].get_weights())(scale0_layer_24)
scale0_layer_26 = ZeroPadding2D((1,1))(scale0_layer_25)
scale0_pool_4 = MaxPooling2D((3,3), strides=(1,1))(scale0_layer_26)
################# scale0_pool_4 = scale0_layer_27 ###############

scale0_layer_28 = ZeroPadding2D((2,2))(scale0_pool_4)
scale0_layer_29 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2),weights = model1.layers[15].get_weights())(scale0_layer_28)
scale0_layer_30 = ZeroPadding2D((2,2))(scale0_layer_29)
scale0_layer_31 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2),weights = model1.layers[16].get_weights())(scale0_layer_30)
scale0_layer_32 = ZeroPadding2D((2,2))(scale0_layer_31)
scale0_layer_33 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2),weights = model1.layers[17].get_weights())(scale0_layer_32)
scale0_layer_34 = ZeroPadding2D((1,1))(scale0_layer_33)
scale0_pool_5 = MaxPooling2D((3,3), strides=(1,1))(scale0_layer_34)
##############scale0_pool_5 = scale0_layer_35################## 

scale0_layer_36 = ZeroPadding2D((12,12))(scale0_pool_5)
scale0_layer_37 = Convolution2D(1024, (3, 3), activation='relu',dilation_rate=(12,12),kernel_initializer='random_uniform')(scale0_layer_36)
scale0_layer_38 = Dropout(rate = 0.5, noise_shape=None, seed=None)(scale0_layer_37)
scale0_layer_39 = Convolution2D(1024, (1, 1), activation='relu',kernel_initializer='random_uniform')(scale0_layer_38)
scale0_fc7 = Dropout(rate= 0.5, noise_shape=None, seed=None)(scale0_layer_39)

########################################################################################33
########################################################################################33
######################################scale1=0.5###########################################33

main_input1 = Input(shape=(80,80,3), dtype='float32', name='main_input1')
scale1_layer_1 = ZeroPadding2D((1,1))(main_input1)
scale1_layer_2 = Convolution2D(64, (3, 3), activation='relu', weights = model2.layers[1].get_weights())(scale1_layer_1)
scale1_layer_3 = ZeroPadding2D((1,1))(scale1_layer_2)
scale1_layer_4 = Convolution2D(64, (3, 3), activation='relu', weights = model2.layers[2].get_weights())(scale1_layer_3)
scale1_layer_5 = ZeroPadding2D((1,1))(scale1_layer_4)
scale1_pool_1 = MaxPooling2D((3,3), strides=(2,2))(scale1_layer_5)
################ scale1_pool_1 = scale1_layer_5 ##################

scale1_layer_6 = ZeroPadding2D((1,1))(scale1_pool_1)
scale1_layer_7 = Convolution2D(128, (3, 3), activation='relu', weights = model2.layers[4].get_weights())(scale1_layer_6)
scale1_layer_8 = ZeroPadding2D((1,1))(scale1_layer_7)
scale1_layer_9 = Convolution2D(128, (3, 3), activation='relu', weights = model2.layers[5].get_weights())(scale1_layer_8)
scale1_layer_10 = ZeroPadding2D((1,1))(scale1_layer_9)
scale1_pool_2 = MaxPooling2D((3,3), strides=(2,2))(scale1_layer_10)
################ scale1_pool_2 = scale1_layer_11 ##################

scale1_layer_12 = ZeroPadding2D((1,1))(scale1_pool_2)
scale1_layer_13 = Convolution2D(256, (3, 3), activation='relu', weights = model2.layers[7].get_weights())(scale1_layer_12)
scale1_layer_14 = ZeroPadding2D((1,1))(scale1_layer_13)
scale1_layer_15 = Convolution2D(256, (3, 3), activation='relu', weights = model2.layers[8].get_weights())(scale1_layer_14)
scale1_layer_16 = ZeroPadding2D((1,1))(scale1_layer_15)
scale1_layer_17 = Convolution2D(256, (3, 3), activation='relu', weights = model2.layers[9].get_weights())(scale1_layer_16)
scale1_layer_18 = ZeroPadding2D((1,1))(scale1_layer_17)
scale1_pool_3 = MaxPooling2D((3,3), strides=(2,2))(scale1_layer_18)
################ scale1_pool_3 = scale1_layer_19 ##################

scale1_layer_20 = ZeroPadding2D((1,1))(scale1_pool_3)
scale1_layer_21 = Convolution2D(512, (3, 3), activation='relu', weights = model2.layers[11].get_weights())(scale1_layer_20)
scale1_layer_22 = ZeroPadding2D((1,1))(scale1_layer_21)
scale1_layer_23 = Convolution2D(512, (3, 3), activation='relu', weights = model2.layers[12].get_weights())(scale1_layer_22)
scale1_layer_24 = ZeroPadding2D((1,1))(scale1_layer_23)
scale1_layer_25 = Convolution2D(512, (3, 3), activation='relu', weights = model2.layers[13].get_weights())(scale1_layer_24)
scale1_layer_26 = ZeroPadding2D((1,1))(scale1_layer_25)
scale1_pool_4 = MaxPooling2D((3,3), strides=(1,1))(scale1_layer_26)
################# scale1_pool_4 = scale1_layer_27 ###############

scale1_layer_28 = ZeroPadding2D((2,2))(scale1_pool_4)
scale1_layer_29 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2), weights = model2.layers[15].get_weights())(scale1_layer_28)
scale1_layer_30 = ZeroPadding2D((2,2))(scale1_layer_29)
scale1_layer_31 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2), weights = model2.layers[16].get_weights())(scale1_layer_30)
scale1_layer_32 = ZeroPadding2D((2,2))(scale1_layer_31)
scale1_layer_33 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2), weights = model2.layers[17].get_weights())(scale1_layer_32)
scale1_layer_34 = ZeroPadding2D((1,1))(scale1_layer_33)
scale1_pool_5 = MaxPooling2D((3,3), strides=(1,1))(scale1_layer_34)
##############scale1_pool_5 = scale1_layer_35################## 

scale1_layer_36 = ZeroPadding2D((12,12))(scale1_pool_5)
scale1_layer_37 = Convolution2D(1024, (3, 3), activation='relu',dilation_rate=(12,12), kernel_initializer='random_uniform')(scale1_layer_36)
scale1_layer_38 = Dropout(rate = 0.5, noise_shape=None, seed=None)(scale1_layer_37)
scale1_layer_39 = Convolution2D(1024, (1, 1), activation='relu', kernel_initializer='random_uniform')(scale1_layer_38)
scale1_fc7 = Dropout(rate= 0.5, noise_shape=None, seed=None)(scale1_layer_39)


########################################################################################33
########################################################################################33
##############################scale2 = 0.75###############################################


main_input2 = Input(shape=(120,120,3), dtype='float32', name='main_input2')
scale2_layer_1 = ZeroPadding2D((1,1))(main_input2)
scale2_layer_2 = Convolution2D(64, (3, 3), activation='relu', weights = model3.layers[1].get_weights())(scale2_layer_1)
scale2_layer_3 = ZeroPadding2D((1,1))(scale2_layer_2)
scale2_layer_4 = Convolution2D(64, (3, 3), activation='relu', weights = model3.layers[2].get_weights())(scale2_layer_3)
scale2_layer_5 = ZeroPadding2D((1,1))(scale2_layer_4)
scale2_pool_1 = MaxPooling2D((3,3), strides=(2,2))(scale2_layer_5)
################ scale2_pool_1 = scale2_layer_5 ##################

scale2_layer_6 = ZeroPadding2D((1,1))(scale2_pool_1)
scale2_layer_7 = Convolution2D(128, (3, 3), activation='relu', weights = model3.layers[4].get_weights())(scale2_layer_6)
scale2_layer_8 = ZeroPadding2D((1,1))(scale2_layer_7)
scale2_layer_9 = Convolution2D(128, (3, 3), activation='relu', weights = model3.layers[5].get_weights())(scale2_layer_8)
scale2_layer_10 = ZeroPadding2D((1,1))(scale2_layer_9)
scale2_pool_2 = MaxPooling2D((3,3), strides=(2,2))(scale2_layer_10)
################ scale2_pool_2 = scale2_layer_11 ##################

scale2_layer_12 = ZeroPadding2D((1,1))(scale2_pool_2)
scale2_layer_13 = Convolution2D(256, (3, 3), activation='relu', weights = model3.layers[7].get_weights())(scale2_layer_12)
scale2_layer_14 = ZeroPadding2D((1,1))(scale2_layer_13)
scale2_layer_15 = Convolution2D(256, (3, 3), activation='relu', weights = model3.layers[8].get_weights())(scale2_layer_14)
scale2_layer_16 = ZeroPadding2D((1,1))(scale2_layer_15)
scale2_layer_17 = Convolution2D(256, (3, 3), activation='relu', weights = model3.layers[9].get_weights())(scale2_layer_16)
scale2_layer_18 = ZeroPadding2D((1,1))(scale2_layer_17)
scale2_pool_3 = MaxPooling2D((3,3), strides=(2,2))(scale2_layer_18)
################ scale2_pool_3 = scale2_layer_19 ##################

scale2_layer_20 = ZeroPadding2D((1,1))(scale2_pool_3)
scale2_layer_21 = Convolution2D(512, (3, 3), activation='relu', weights = model3.layers[11].get_weights())(scale2_layer_20)
scale2_layer_22 = ZeroPadding2D((1,1))(scale2_layer_21)
scale2_layer_23 = Convolution2D(512, (3, 3), activation='relu', weights = model3.layers[12].get_weights())(scale2_layer_22)
scale2_layer_24 = ZeroPadding2D((1,1))(scale2_layer_23)
scale2_layer_25 = Convolution2D(512, (3, 3), activation='relu', weights = model3.layers[13].get_weights())(scale2_layer_24)
scale2_layer_26 = ZeroPadding2D((1,1))(scale2_layer_25)
scale2_pool_4 = MaxPooling2D((3,3), strides=(1,1))(scale2_layer_26)
################# scale2_pool_4 = scale2_layer_27 ###############

scale2_layer_28 = ZeroPadding2D((2,2))(scale2_pool_4)
scale2_layer_29 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2), weights = model3.layers[15].get_weights())(scale2_layer_28)
scale2_layer_30 = ZeroPadding2D((2,2))(scale2_layer_29)
scale2_layer_31 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2), weights = model3.layers[16].get_weights())(scale2_layer_30)
scale2_layer_32 = ZeroPadding2D((2,2))(scale2_layer_31)
scale2_layer_33 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2), weights = model3.layers[17].get_weights())(scale2_layer_32)
scale2_layer_34 = ZeroPadding2D((1,1))(scale2_layer_33)
scale2_pool_5 = MaxPooling2D((3,3), strides=(1,1))(scale2_layer_34)
##############scale2_pool_5 = scale2_layer_35################## 

scale2_layer_36 = ZeroPadding2D((12,12))(scale2_pool_5)
scale2_layer_37 = Convolution2D(1024, (3, 3), activation='relu',dilation_rate=(12,12),kernel_initializer='random_uniform')(scale2_layer_36)
scale2_layer_38 = Dropout(rate = 0.5, noise_shape=None, seed=None)(scale2_layer_37)
scale2_layer_39 = Convolution2D(1024, (1, 1), activation='relu', kernel_initializer='random_uniform')(scale2_layer_38)
scale2_fc7 = Dropout(rate= 0.5, noise_shape=None, seed=None)(scale2_layer_39)

scale0_mask_1 = Convolution2D(64, (1, 1), strides=(1,1),activation='relu', kernel_initializer='random_uniform')(scale0_fc7)
scale0_mask_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale0_skip_1 = ZeroPadding2D((1,1))(scale0_fc7)
scale0_skip_1 = Convolution2D(64, (3, 3),activation='relu', kernel_initializer='random_uniform')(scale0_skip_1)
scale0_skip_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##############################################################
scale0_concat_1 = Concatenate()([scale0_skip_1, scale0_mask_1])
scale0_concat_1 = ZeroPadding2D((1,1))(scale0_concat_1)
scale0_mask_2 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_concat_1)
scale0_mask_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale0_skip_2 = ZeroPadding2D((1,1))(scale0_pool_5)
scale0_skip_2 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_skip_2)
scale0_skip_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###################################################3
scale0_concat_2 = Concatenate()([scale0_skip_2, scale0_mask_2])
scale0_mask_3 = ZeroPadding2D((1,1))(scale0_concat_2)
scale0_mask_3 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_mask_3)
scale0_mask_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale0_skip_3 = ZeroPadding2D((1,1))(scale0_pool_4)
scale0_skip_3 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_skip_3)
scale0_skip_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##########################################333
scale0_concat_3 = Concatenate()([scale0_skip_3, scale0_mask_3])
scale0_mask_4 = ZeroPadding2D((1,1))(scale0_concat_3)
scale0_mask_4 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_mask_4)
scale0_mask_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
## upsampling to match dimensions 

scale0_skip_4 = ZeroPadding2D((1,1))(scale0_pool_3)
scale0_skip_4 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_skip_4)
scale0_skip_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

############################################################
scale0_concat_4 = Concatenate()([scale0_skip_4, scale0_mask_4])
scale0_mask_5 = ZeroPadding2D((1,1))(scale0_concat_4)
scale0_mask_5 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_mask_5)
scale0_mask_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale0_mask_5_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu', kernel_initializer='random_uniform')(scale0_mask_5)
# scale0_mask_5_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)
#(scale0_mask_5)

scale0_skip_5 = ZeroPadding2D((1,1))(scale0_pool_2)
scale0_skip_5 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_skip_5)
scale0_skip_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###########################################################
scale0_concat_5 = Concatenate()([scale0_skip_5, scale0_mask_5_up])
scale0_mask_6 = ZeroPadding2D((1,1))(scale0_concat_5)
scale0_mask_6 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_mask_6)
scale0_mask_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale0_mask_6_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu', kernel_initializer='random_uniform')(scale0_mask_6)

scale0_skip_6 = ZeroPadding2D((1,1))(scale0_pool_1)
scale0_skip_6 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_skip_6)
scale0_skip_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###############################################
scale0_concat_6 = Concatenate()([scale0_skip_6, scale0_mask_6_up])
scale0_mask_7 = ZeroPadding2D((1,1))(scale0_concat_6)
scale0_mask_7 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale0_mask_7)
scale0_mask_7 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_7)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale0_mask_7_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu', kernel_initializer='random_uniform')(scale0_mask_7)


# model0 = Model(inputs = main_input0,outputs = scale0_mask_7_up)
# print(model0.summary())





scale1_mask_1 = Convolution2D(64, (1, 1), strides=(1,1),activation='relu', kernel_initializer='random_uniform')(scale1_fc7)
scale1_mask_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale1_skip_1 = ZeroPadding2D((1,1))(scale1_fc7)
scale1_skip_1 = Convolution2D(64, (3, 3),activation='relu', kernel_initializer='random_uniform')(scale1_skip_1)
scale1_skip_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##############################################################
scale1_concat_1 = Concatenate()([scale1_skip_1, scale1_mask_1])
scale1_concat_1 = ZeroPadding2D((1,1))(scale1_concat_1)
scale1_mask_2 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_concat_1)
scale1_mask_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale1_skip_2 = ZeroPadding2D((1,1))(scale1_pool_5)
scale1_skip_2 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_skip_2)
scale1_skip_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###################################################3
scale1_concat_2 = Concatenate()([scale1_skip_2, scale1_mask_2])
scale1_mask_3 = ZeroPadding2D((1,1))(scale1_concat_2)
scale1_mask_3 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_mask_3)
scale1_mask_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale1_skip_3 = ZeroPadding2D((1,1))(scale1_pool_4)
scale1_skip_3 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_skip_3)
scale1_skip_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##########################################333
scale1_concat_3 = Concatenate()([scale1_skip_3, scale1_mask_3])
scale1_mask_4 = ZeroPadding2D((1,1))(scale1_concat_3)
scale1_mask_4 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_mask_4)
scale1_mask_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
# scale1_mask_4_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale1_mask_4)
## upsampling to match dimensions 

scale1_skip_4 = ZeroPadding2D((1,1))(scale1_pool_3)
scale1_skip_4 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_skip_4)
scale1_skip_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

############################################################
scale1_concat_4 = Concatenate()([scale1_skip_4, scale1_mask_4])
scale1_mask_5 = ZeroPadding2D((1,1))(scale1_concat_4)
scale1_mask_5 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_mask_5)
scale1_mask_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale1_mask_5_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu', kernel_initializer='random_uniform')(scale1_mask_5)
# scale1_mask_5_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale1_mask_5)

scale1_skip_5 = ZeroPadding2D((1,1))(scale1_pool_2)
scale1_skip_5 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_skip_5)
scale1_skip_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###########################################################
scale1_concat_5 = Concatenate()([scale1_skip_5, scale1_mask_5_up])
scale1_mask_6 = ZeroPadding2D((1,1))(scale1_concat_5)
scale1_mask_6 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_mask_6)
scale1_mask_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale1_mask_6_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale1_mask_6)

scale1_skip_6 = ZeroPadding2D((1,1))(scale1_pool_1)
scale1_skip_6 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_skip_6)
scale1_skip_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###############################################
scale1_concat_6 = Concatenate()([scale1_skip_6, scale1_mask_6_up])
scale1_mask_7 = ZeroPadding2D((1,1))(scale1_concat_6)
scale1_mask_7 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform')(scale1_mask_7)
scale1_mask_7 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_7)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale1_mask_7_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale1_mask_7)


# model1 = Model(inputs = main_input1,outputs = scale1_mask_7_up)
# print(model1.summary())




scale2_mask_1 = Convolution2D(64, (1, 1), strides=(1,1),activation='relu',kernel_initializer='random_uniform')(scale2_fc7)
scale2_mask_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale2_skip_1 = ZeroPadding2D((1,1))(scale2_fc7)
scale2_skip_1 = Convolution2D(64, (3, 3),activation='relu',kernel_initializer='random_uniform')(scale2_skip_1)
scale2_skip_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##############################################################
scale2_concat_1 = Concatenate()([scale2_skip_1, scale2_mask_1])
scale2_concat_1 = ZeroPadding2D((1,1))(scale2_concat_1)
scale2_mask_2 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_concat_1)
scale2_mask_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale2_skip_2 = ZeroPadding2D((1,1))(scale2_pool_5)
scale2_skip_2 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_skip_2)
scale2_skip_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###################################################3
scale2_concat_2 = Concatenate()([scale2_skip_2, scale2_mask_2])
scale2_mask_3 = ZeroPadding2D((1,1))(scale2_concat_2)
scale2_mask_3 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_mask_3)
scale2_mask_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale2_skip_3 = ZeroPadding2D((1,1))(scale2_pool_4)
scale2_skip_3 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_skip_3)
scale2_skip_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##########################################333
scale2_concat_3 = Concatenate()([scale2_skip_3, scale2_mask_3])
scale2_mask_4 = ZeroPadding2D((1,1))(scale2_concat_3)
scale2_mask_4 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_mask_4)
scale2_mask_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
# scale2_mask_4_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale2_mask_4)
## upsampling to match dimensions 

scale2_skip_4 = ZeroPadding2D((1,1))(scale2_pool_3)
scale2_skip_4 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_skip_4)
scale2_skip_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

############################################################
scale2_concat_4 = Concatenate()([scale2_skip_4, scale2_mask_4])
scale2_mask_5 = ZeroPadding2D((1,1))(scale2_concat_4)
scale2_mask_5 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_mask_5)
scale2_mask_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale2_mask_5_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu',kernel_initializer='random_uniform')(scale2_mask_5)
# scale2_mask_5_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale2_mask_5)

scale2_skip_5 = ZeroPadding2D((1,1))(scale2_pool_2)
scale2_skip_5 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_skip_5)
scale2_skip_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###########################################################
scale2_concat_5 = Concatenate()([scale2_skip_5, scale2_mask_5_up])
scale2_mask_6 = ZeroPadding2D((1,1))(scale2_concat_5)
scale2_mask_6 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_mask_6)
scale2_mask_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale2_mask_6_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu',kernel_initializer='random_uniform')(scale2_mask_6)

scale2_skip_6 = ZeroPadding2D((1,1))(scale2_pool_1)
scale2_skip_6 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_skip_6)
scale2_skip_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###############################################
scale2_concat_6 = Concatenate()([scale2_skip_6, scale2_mask_6_up])
scale2_mask_7 = ZeroPadding2D((1,1))(scale2_concat_6)
scale2_mask_7 = Convolution2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_mask_7)
scale2_mask_7 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_7)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale2_mask_7_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu',kernel_initializer='random_uniform')(scale2_mask_7)


# model2 = Model(inputs = main_input0,outputs = scale0_fc7)
# print(model2.summary())

########################################################################################33
########################################################################################33
########################################################################################33
########################## Upsampling Ouputs to Same size ###################################


scale0_output_mask= ZeroPadding2D((1,1))(scale0_mask_7_up)
scale0_output_mask = Convolution2D(1, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale0_output_mask)
scale0_output_mask = Dense(1,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_output_mask)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale1_output_mask= ZeroPadding2D((1,1))(scale1_mask_7_up)
scale1_output_mask = Convolution2D(1, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale1_output_mask)
scale1_output_mask = Dense(1,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_output_mask)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale2_output_mask= ZeroPadding2D((1,1))(scale2_mask_7_up)
scale2_output_mask = Convolution2D(1, (3, 3), activation='relu',kernel_initializer='random_uniform')(scale2_output_mask)
scale2_output_mask = Dense(1,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_output_mask)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)


scale1_mask_7_up_interpol = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale1_mask_7_up)

scale2_mask_7_up_interpol = keras.layers.UpSampling2D(size=(4, 4),interpolation = 'bilinear',data_format=None)(scale2_mask_7_up)
scale2_mask_7_up_interpol = keras.layers.AveragePooling2D(pool_size=(3, 3),data_format=None)(scale2_mask_7_up_interpol)

scale1_output_mask_interpol = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale1_output_mask)

scale2_output_mask_interpol = keras.layers.UpSampling2D(size=(4, 4),interpolation = 'bilinear',data_format=None)(scale2_output_mask)
scale2_output_mask_interpol = keras.layers.AveragePooling2D(pool_size=(3, 3),data_format=None)(scale2_output_mask_interpol)


########################################################################################33
########################################################################################33
########################################################################################33
########################## Attention Module ###################################

mask_7_up_concat = Concatenate()([scale0_mask_7_up, scale1_mask_7_up_interpol, scale2_mask_7_up_interpol])

att_conv1= ZeroPadding2D((1,1))(mask_7_up_concat)
att_conv1 = Convolution2D(512, (3, 3), activation='relu',kernel_initializer='random_uniform')(att_conv1)
att_conv1 = Dropout(rate = 0.5, noise_shape=None, seed=None)(att_conv1)

attention = Convolution2D(3, (1, 1), activation='softmax',kernel_initializer='random_uniform')(att_conv1)

attention0 = Lambda(lambda attention : attention[:,:,:,0])(attention)
attention1 = Lambda(lambda attention : attention[:,:,:,1])(attention)
attention2 = Lambda(lambda attention : attention[:,:,:,2])(attention)
# scale0_output_mask = squeeze(scale0_output_mask,3)
# attention1 = Lambda(lambda x : expand_dims(x,axis=-1))(attention1)
# scale0_output_mask_product = expand_dims(scale0_output_mask_product,axis=-1)

scale0_output_mask_sq0 = Lambda(lambda x : squeeze(x,axis=3))(scale0_output_mask)
scale0_output_mask_product = keras.layers.Multiply()([scale0_output_mask_sq0,attention0])


scale1_output_mask_interpol_sq1 = Lambda(lambda x : squeeze(x,axis=3))(scale1_output_mask_interpol)
scale1_output_mask_product = keras.layers.Multiply()([scale1_output_mask_interpol_sq1,attention1])


scale2_output_mask_interpol_sq2 = Lambda(lambda x : squeeze(x,axis=3))(scale2_output_mask_interpol)
scale2_output_mask_product = keras.layers.Multiply()([scale2_output_mask_interpol_sq2,attention2])

output_fusion= keras.layers.Add()([scale0_output_mask_product,scale1_output_mask_product,scale2_output_mask_product])


########################################################################################33
########################################################################################33
########################################################################################33
########################## Classifier for each resolution###################################

scale0_fc8 = Convolution2D(1, (1, 1), activation=None,kernel_initializer='random_uniform')(scale0_fc7)
scale0_fc8 = Dense(1,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_fc8)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale0_fc8 = Lambda(lambda x : squeeze(x,axis=3))(scale0_fc8)


scale1_fc8 = Convolution2D(1, (1, 1), activation=None,kernel_initializer='random_uniform')(scale1_fc7)
scale1_fc8 = Dense(1,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_fc8)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
# scale1_fc8 = Lambda(lambda x : squeeze(x,axis=3))(scale1_fc8)


scale2_fc8 = Convolution2D(1, (1, 1), activation=None,kernel_initializer='random_uniform')(scale2_fc7)
scale2_fc8 = Dense(1,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_fc8)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
# scale2_fc8 = Lambda(lambda x : squeeze(x,axis=3))(scale2_fc8)



########################################upsampling###############################################

scale1_fc7_interp = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale1_fc7)

scale2_fc7_interp = keras.layers.UpSampling2D(size=(4, 4),interpolation = 'bilinear',data_format=None)(scale2_fc7)
scale2_fc7_interp = keras.layers.AveragePooling2D(pool_size=(3, 3),data_format=None)(scale2_fc7_interp)


scale1_fc8_interp = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale1_fc8)
scale1_fc8_interp = Lambda(lambda x : squeeze(x,axis=3))(scale1_fc8_interp)

scale2_fc8_interp = keras.layers.UpSampling2D(size=(4, 4),interpolation = 'bilinear',data_format=None)(scale2_fc8)
scale2_fc8_interp = keras.layers.AveragePooling2D(pool_size=(3, 3),data_format=None)(scale2_fc8_interp)
scale2_fc8_interp = Lambda(lambda x : squeeze(x,axis=3))(scale2_fc8_interp	)


#################fc7 concat and attention module###################################################################3

fc7_concat = Concatenate()([scale0_fc7,scale1_fc7_interp,scale2_fc7_interp])

att_conv1_fc7 = ZeroPadding2D((1,1))(fc7_concat)
att_conv1_fc7 = Convolution2D(512, (3, 3), activation='relu',kernel_initializer='random_uniform')(att_conv1_fc7)
att_conv1_fc7 = Dropout(rate = 0.5, noise_shape=None, seed=None)(att_conv1_fc7)

attention_fc7 = Convolution2D(3, (1, 1), activation='softmax',kernel_initializer='random_uniform')(att_conv1_fc7)
# print(scale1_fc8.shape)

attention0_fc7 = Lambda(lambda attention : attention[:,:,:,0])(attention_fc7)
attention1_fc7 = Lambda(lambda attention : attention[:,:,:,1])(attention_fc7)
attention2_fc7 = Lambda(lambda attention : attention[:,:,:,2])(attention_fc7)

scale0_fc8_product = keras.layers.Multiply()([scale0_fc8,attention0_fc7])
scale1_fc8_product = keras.layers.Multiply()([scale1_fc8_interp,attention1_fc7])
scale2_fc8_product = keras.layers.Multiply()([scale2_fc8_interp,attention2_fc7])

fc8_fusion= keras.layers.Add()([scale0_fc8_product,scale1_fc8_product,scale2_fc8_product])

##############################################combining fc8 and mask7_up#####################3#
fc8_fusion = Lambda(lambda x : expand_dims(x,axis=-1))(fc8_fusion)
fc8_fusion_interp = keras.layers.UpSampling2D(size=(8, 8),interpolation = 'bilinear',data_format=None)(fc8_fusion)
# print(fc8_fusion_interp.shape)
fc8_fusion_interp = Lambda(lambda x : squeeze(x,axis=3))(fc8_fusion_interp)

final_fusion= keras.layers.Add()([fc8_fusion_interp,output_fusion])

final_fusion_softmax= Lambda(lambda x :keras.activations.sigmoid(x))(final_fusion)


model = Model(inputs = [main_input0,main_input1,main_input2],outputs = final_fusion_softmax)
print(model.summary())


# In[5]:


import random
imgs_dir = './mix_data'
ground_truth = './mix_data_mask'
imageNames = []
groundTruthNames = []
for filename in os.listdir(imgs_dir):
    filepath = imgs_dir + '/' +filename
    imageNames.append(filepath)
for filename in os.listdir(ground_truth):
    filepath = ground_truth + '/' +filename
    groundTruthNames.append(filepath)
imageNames = sorted(imageNames)
groundTruthNames = sorted(groundTruthNames)
def train_generator():
    while True:
        batch_size = 4
        images0 = []
        images1 = []
        images2 = []
        y = []
        for i in range(batch_size):
            idx = random.randint(1, 3000)
            img = cv2.imread(imageNames[idx])
            resized_image_0 = cv2.resize(img, (160, 160))
            resized_image_1 = cv2.resize(img, (80, 80))
            resized_image_2 = cv2.resize(img, (120, 120))
            images0.append(resized_image_0/255)
            images1.append(resized_image_1/255)
            images2.append(resized_image_2/255)
            img = cv2.imread(groundTruthNames[idx], 0)
            res = cv2.resize(img, (160, 160))
            res = res/255
            for i in range(160):
                for j in range(160):
                    if(res[i][j]>0.5):
                        res[i][j] = 1
                    else:
                        res[i][j] = 0
            y.append(res)
        images0 = np.array(images0).reshape((batch_size, 160, 160, 3))
        images1 = np.array(images1).reshape((batch_size, 80, 80, 3))
        images2 = np.array(images2).reshape((batch_size, 120, 120, 3))
        y = np.array(y).reshape((batch_size, 160, 160))
        yield [images0, images1, images2], y

def val_generator():
    while True:
        batch_size = 4
        images0 = []
        images1 = []
        images2 = []
        y = []
        for i in range(batch_size):
            idx = random.randint(3000, 4000)
            img = cv2.imread(imageNames[idx])
            resized_image_0 = cv2.resize(img, (160, 160))
            resized_image_1 = cv2.resize(img, (80, 80))
            resized_image_2 = cv2.resize(img, (120, 120))
            images0.append(resized_image_0/255)
            images1.append(resized_image_1/255)
            images2.append(resized_image_2/255)
            img = cv2.imread(groundTruthNames[idx], 0)
            res = cv2.resize(img, (160, 160))
            res = res/255
            for i in range(160):
                for j in range(160):
                    if(res[i][j]>0.5):
                        res[i][j] = 1
                    else:
                        res[i][j] = 0
            y.append(res)
        images0 = np.array(images0).reshape((batch_size, 160,160,3))
        images1 = np.array(images1).reshape((batch_size, 80, 80,3))
        images2 = np.array(images2).reshape((batch_size, 120, 120,3))
        y = np.array(y).reshape((batch_size, 160, 160))
        yield [images0, images1, images2], y
        
        
    
    


# In[19]:


adam = Adam(lr=0.00001)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[20]:


mcp_save = ModelCheckpoint('weights.hdf5', save_best_only=True, monitor='val_loss', mode='min', period = 5)


# In[22]:


model.fit_generator(train_generator(),steps_per_epoch = 1000, validation_data=val_generator(), validation_steps = 10, epochs = 100,callbacks=[mcp_save])


# In[16]:


img = cv2.imread(imageNames[501])
plt.imshow(img)
plt.show()
resized_image_0 = cv2.resize(img, (160,160))
resized_image_1 = cv2.resize(img, (80,80))
resized_image_2 = cv2.resize(img, (120,120))
out = model.predict([np.array(resized_image_0).reshape((1, 160, 160,3)), np.array(resized_image_1).reshape((1,80, 80,3)), np.array(resized_image_2).reshape((1,120,120,3))])


# In[17]:


out[0]


# In[18]:



plt.imshow(out[0])
                


