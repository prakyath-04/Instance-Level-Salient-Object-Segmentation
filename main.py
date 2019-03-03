from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers import Input, Concatenate, Conv2DTranspose, Lambda
from keras.utils import plot_model
from keras.backend import dot

#from tensorflow.python.keras.models import Model

import keras
import cv2
import numpy as np

# def VGG(weight_path=None):
model0 = Sequential()
model1 = Sequential()
model2 = Sequential()

########################################################################################33
########################################################################################33
########################################################################################33

main_input0 = Input(shape=(320,320,3), dtype='float32', name='main_input0')
scale0_layer_1 = ZeroPadding2D((1,1))(main_input0)
scale0_layer_2 = Convolution2D(64, (3, 3), activation='relu')(scale0_layer_1)
scale0_layer_3 = ZeroPadding2D((1,1))(scale0_layer_2)
scale0_layer_4 = Convolution2D(64, (3, 3), activation='relu')(scale0_layer_3)
scale0_layer_5 = ZeroPadding2D((1,1))(scale0_layer_4)
scale0_pool_1 = MaxPooling2D((3,3), strides=(2,2))(scale0_layer_5)
################ scale0_pool_1 = scale0_layer_5 ##################

scale0_layer_6 = ZeroPadding2D((1,1))(scale0_pool_1)
scale0_layer_7 = Convolution2D(128, (3, 3), activation='relu')(scale0_layer_6)
scale0_layer_8 = ZeroPadding2D((1,1))(scale0_layer_7)
scale0_layer_9 = Convolution2D(128, (3, 3), activation='relu')(scale0_layer_8)
scale0_layer_10 = ZeroPadding2D((1,1))(scale0_layer_9)
scale0_pool_2 = MaxPooling2D((3,3), strides=(2,2))(scale0_layer_10)
################ scale0_pool_2 = scale0_layer_11 ##################

scale0_layer_12 = ZeroPadding2D((1,1))(scale0_pool_2)
scale0_layer_13 = Convolution2D(256, (3, 3), activation='relu')(scale0_layer_12)
scale0_layer_14 = ZeroPadding2D((1,1))(scale0_layer_13)
scale0_layer_15 = Convolution2D(256, (3, 3), activation='relu')(scale0_layer_14)
scale0_layer_16 = ZeroPadding2D((1,1))(scale0_layer_15)
scale0_layer_17 = Convolution2D(256, (3, 3), activation='relu')(scale0_layer_16)
scale0_layer_18 = ZeroPadding2D((1,1))(scale0_layer_17)
scale0_pool_3 = MaxPooling2D((3,3), strides=(2,2))(scale0_layer_18)
################ scale0_pool_3 = scale0_layer_19 ##################

scale0_layer_20 = ZeroPadding2D((1,1))(scale0_pool_3)
scale0_layer_21 = Convolution2D(512, (3, 3), activation='relu')(scale0_layer_20)
scale0_layer_22 = ZeroPadding2D((1,1))(scale0_layer_21)
scale0_layer_23 = Convolution2D(512, (3, 3), activation='relu')(scale0_layer_22)
scale0_layer_24 = ZeroPadding2D((1,1))(scale0_layer_23)
scale0_layer_25 = Convolution2D(512, (3, 3), activation='relu')(scale0_layer_24)
scale0_layer_26 = ZeroPadding2D((1,1))(scale0_layer_25)
scale0_pool_4 = MaxPooling2D((3,3), strides=(1,1))(scale0_layer_26)
################# scale0_pool_4 = scale0_layer_27 ###############

scale0_layer_28 = ZeroPadding2D((2,2))(scale0_pool_4)
scale0_layer_29 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2))(scale0_layer_28)
scale0_layer_30 = ZeroPadding2D((2,2))(scale0_layer_29)
scale0_layer_31 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2))(scale0_layer_30)
scale0_layer_32 = ZeroPadding2D((2,2))(scale0_layer_31)
scale0_layer_33 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2))(scale0_layer_32)
scale0_layer_34 = ZeroPadding2D((1,1))(scale0_layer_33)
scale0_pool_5 = MaxPooling2D((3,3), strides=(1,1))(scale0_layer_34)
##############scale0_pool_5 = scale0_layer_35################## 

scale0_layer_36 = ZeroPadding2D((12,12))(scale0_pool_5)
scale0_layer_37 = Convolution2D(1024, (3, 3), activation='relu',dilation_rate=(12,12))(scale0_layer_36)
scale0_layer_38 = Dropout(rate = 0.5, noise_shape=None, seed=None)(scale0_layer_37)
scale0_layer_39 = Convolution2D(1024, (1, 1), activation='relu')(scale0_layer_38)
scale0_fc7 = Dropout(rate= 0.5, noise_shape=None, seed=None)(scale0_layer_39)

scale0_mask_1 = Convolution2D(64, (1, 1), strides=(1,1),activation='relu')(scale0_fc7)
scale0_mask_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale0_skip_1 = ZeroPadding2D((1,1))(scale0_fc7)
scale0_skip_1 = Convolution2D(64, (3, 3),activation='relu')(scale0_skip_1)
scale0_skip_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##############################################################
scale0_concat_1 = Concatenate()([scale0_skip_1, scale0_mask_1])
scale0_concat_1 = ZeroPadding2D((1,1))(scale0_concat_1)
scale0_mask_2 = Convolution2D(64, (3, 3), activation='relu')(scale0_concat_1)
scale0_mask_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale0_skip_2 = ZeroPadding2D((1,1))(scale0_pool_5)
scale0_skip_2 = Convolution2D(64, (3, 3), activation='relu')(scale0_skip_2)
scale0_skip_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###################################################3
scale0_concat_2 = Concatenate()([scale0_skip_2, scale0_mask_2])
scale0_mask_3 = ZeroPadding2D((1,1))(scale0_concat_2)
scale0_mask_3 = Convolution2D(64, (3, 3), activation='relu')(scale0_mask_3)
scale0_mask_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale0_skip_3 = ZeroPadding2D((1,1))(scale0_pool_4)
scale0_skip_3 = Convolution2D(64, (3, 3), activation='relu')(scale0_skip_3)
scale0_skip_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##########################################333
scale0_concat_3 = Concatenate()([scale0_skip_3, scale0_mask_3])
scale0_mask_4 = ZeroPadding2D((1,1))(scale0_concat_3)
scale0_mask_4 = Convolution2D(64, (3, 3), activation='relu')(scale0_mask_4)
scale0_mask_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
# scale0_mask_4_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale0_mask_4)
## upsampling to match dimensions 

scale0_skip_4 = ZeroPadding2D((1,1))(scale0_pool_3)
scale0_skip_4 = Convolution2D(64, (3, 3), activation='relu')(scale0_skip_4)
scale0_skip_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

############################################################
scale0_concat_4 = Concatenate()([scale0_skip_4, scale0_mask_4])
scale0_mask_5 = ZeroPadding2D((1,1))(scale0_concat_4)
scale0_mask_5 = Convolution2D(64, (3, 3), activation='relu')(scale0_mask_5)
scale0_mask_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale0_mask_5_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale0_mask_5)
# scale0_mask_5_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale0_mask_5)

scale0_skip_5 = ZeroPadding2D((1,1))(scale0_pool_2)
scale0_skip_5 = Convolution2D(64, (3, 3), activation='relu')(scale0_skip_5)
scale0_skip_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###########################################################
scale0_concat_5 = Concatenate()([scale0_skip_5, scale0_mask_5_up])
scale0_mask_6 = ZeroPadding2D((1,1))(scale0_concat_5)
scale0_mask_6 = Convolution2D(64, (3, 3), activation='relu')(scale0_mask_6)
scale0_mask_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale0_mask_6_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale0_mask_6)

scale0_skip_6 = ZeroPadding2D((1,1))(scale0_pool_1)
scale0_skip_6 = Convolution2D(64, (3, 3), activation='relu')(scale0_skip_6)
scale0_skip_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_skip_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###############################################
scale0_concat_6 = Concatenate()([scale0_skip_6, scale0_mask_6_up])
scale0_mask_7 = ZeroPadding2D((1,1))(scale0_concat_6)
scale0_mask_7 = Convolution2D(64, (3, 3), activation='relu')(scale0_mask_7)
scale0_mask_7 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_mask_7)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale0_mask_7_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale0_mask_7)


# model0 = Model(inputs = main_input0,outputs = scale0_mask_7_up)
# print(model0.summary())



########################################################################################33
########################################################################################33
######################################scale1=0.5###########################################33

main_input1 = Input(shape=(160,160,3), dtype='float32', name='main_input1')
scale1_layer_1 = ZeroPadding2D((1,1))(main_input1)
scale1_layer_2 = Convolution2D(64, (3, 3), activation='relu')(scale1_layer_1)
scale1_layer_3 = ZeroPadding2D((1,1))(scale1_layer_2)
scale1_layer_4 = Convolution2D(64, (3, 3), activation='relu')(scale1_layer_3)
scale1_layer_5 = ZeroPadding2D((1,1))(scale1_layer_4)
scale1_pool_1 = MaxPooling2D((3,3), strides=(2,2))(scale1_layer_5)
################ scale1_pool_1 = scale1_layer_5 ##################

scale1_layer_6 = ZeroPadding2D((1,1))(scale1_pool_1)
scale1_layer_7 = Convolution2D(128, (3, 3), activation='relu')(scale1_layer_6)
scale1_layer_8 = ZeroPadding2D((1,1))(scale1_layer_7)
scale1_layer_9 = Convolution2D(128, (3, 3), activation='relu')(scale1_layer_8)
scale1_layer_10 = ZeroPadding2D((1,1))(scale1_layer_9)
scale1_pool_2 = MaxPooling2D((3,3), strides=(2,2))(scale1_layer_10)
################ scale1_pool_2 = scale1_layer_11 ##################

scale1_layer_12 = ZeroPadding2D((1,1))(scale1_pool_2)
scale1_layer_13 = Convolution2D(256, (3, 3), activation='relu')(scale1_layer_12)
scale1_layer_14 = ZeroPadding2D((1,1))(scale1_layer_13)
scale1_layer_15 = Convolution2D(256, (3, 3), activation='relu')(scale1_layer_14)
scale1_layer_16 = ZeroPadding2D((1,1))(scale1_layer_15)
scale1_layer_17 = Convolution2D(256, (3, 3), activation='relu')(scale1_layer_16)
scale1_layer_18 = ZeroPadding2D((1,1))(scale1_layer_17)
scale1_pool_3 = MaxPooling2D((3,3), strides=(2,2))(scale1_layer_18)
################ scale1_pool_3 = scale1_layer_19 ##################

scale1_layer_20 = ZeroPadding2D((1,1))(scale1_pool_3)
scale1_layer_21 = Convolution2D(512, (3, 3), activation='relu')(scale1_layer_20)
scale1_layer_22 = ZeroPadding2D((1,1))(scale1_layer_21)
scale1_layer_23 = Convolution2D(512, (3, 3), activation='relu')(scale1_layer_22)
scale1_layer_24 = ZeroPadding2D((1,1))(scale1_layer_23)
scale1_layer_25 = Convolution2D(512, (3, 3), activation='relu')(scale1_layer_24)
scale1_layer_26 = ZeroPadding2D((1,1))(scale1_layer_25)
scale1_pool_4 = MaxPooling2D((3,3), strides=(1,1))(scale1_layer_26)
################# scale1_pool_4 = scale1_layer_27 ###############

scale1_layer_28 = ZeroPadding2D((2,2))(scale1_pool_4)
scale1_layer_29 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2))(scale1_layer_28)
scale1_layer_30 = ZeroPadding2D((2,2))(scale1_layer_29)
scale1_layer_31 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2))(scale1_layer_30)
scale1_layer_32 = ZeroPadding2D((2,2))(scale1_layer_31)
scale1_layer_33 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2))(scale1_layer_32)
scale1_layer_34 = ZeroPadding2D((1,1))(scale1_layer_33)
scale1_pool_5 = MaxPooling2D((3,3), strides=(1,1))(scale1_layer_34)
##############scale1_pool_5 = scale1_layer_35################## 

scale1_layer_36 = ZeroPadding2D((12,12))(scale1_pool_5)
scale1_layer_37 = Convolution2D(1024, (3, 3), activation='relu',dilation_rate=(12,12))(scale1_layer_36)
scale1_layer_38 = Dropout(rate = 0.5, noise_shape=None, seed=None)(scale1_layer_37)
scale1_layer_39 = Convolution2D(1024, (1, 1), activation='relu')(scale1_layer_38)
scale1_fc7 = Dropout(rate= 0.5, noise_shape=None, seed=None)(scale1_layer_39)

scale1_mask_1 = Convolution2D(64, (1, 1), strides=(1,1),activation='relu')(scale1_fc7)
scale1_mask_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale1_skip_1 = ZeroPadding2D((1,1))(scale1_fc7)
scale1_skip_1 = Convolution2D(64, (3, 3),activation='relu')(scale1_skip_1)
scale1_skip_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##############################################################
scale1_concat_1 = Concatenate()([scale1_skip_1, scale1_mask_1])
scale1_concat_1 = ZeroPadding2D((1,1))(scale1_concat_1)
scale1_mask_2 = Convolution2D(64, (3, 3), activation='relu')(scale1_concat_1)
scale1_mask_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale1_skip_2 = ZeroPadding2D((1,1))(scale1_pool_5)
scale1_skip_2 = Convolution2D(64, (3, 3), activation='relu')(scale1_skip_2)
scale1_skip_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###################################################3
scale1_concat_2 = Concatenate()([scale1_skip_2, scale1_mask_2])
scale1_mask_3 = ZeroPadding2D((1,1))(scale1_concat_2)
scale1_mask_3 = Convolution2D(64, (3, 3), activation='relu')(scale1_mask_3)
scale1_mask_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale1_skip_3 = ZeroPadding2D((1,1))(scale1_pool_4)
scale1_skip_3 = Convolution2D(64, (3, 3), activation='relu')(scale1_skip_3)
scale1_skip_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##########################################333
scale1_concat_3 = Concatenate()([scale1_skip_3, scale1_mask_3])
scale1_mask_4 = ZeroPadding2D((1,1))(scale1_concat_3)
scale1_mask_4 = Convolution2D(64, (3, 3), activation='relu')(scale1_mask_4)
scale1_mask_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
# scale1_mask_4_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale1_mask_4)
## upsampling to match dimensions 

scale1_skip_4 = ZeroPadding2D((1,1))(scale1_pool_3)
scale1_skip_4 = Convolution2D(64, (3, 3), activation='relu')(scale1_skip_4)
scale1_skip_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

############################################################
scale1_concat_4 = Concatenate()([scale1_skip_4, scale1_mask_4])
scale1_mask_5 = ZeroPadding2D((1,1))(scale1_concat_4)
scale1_mask_5 = Convolution2D(64, (3, 3), activation='relu')(scale1_mask_5)
scale1_mask_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale1_mask_5_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale1_mask_5)
# scale1_mask_5_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale1_mask_5)

scale1_skip_5 = ZeroPadding2D((1,1))(scale1_pool_2)
scale1_skip_5 = Convolution2D(64, (3, 3), activation='relu')(scale1_skip_5)
scale1_skip_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###########################################################
scale1_concat_5 = Concatenate()([scale1_skip_5, scale1_mask_5_up])
scale1_mask_6 = ZeroPadding2D((1,1))(scale1_concat_5)
scale1_mask_6 = Convolution2D(64, (3, 3), activation='relu')(scale1_mask_6)
scale1_mask_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale1_mask_6_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale1_mask_6)

scale1_skip_6 = ZeroPadding2D((1,1))(scale1_pool_1)
scale1_skip_6 = Convolution2D(64, (3, 3), activation='relu')(scale1_skip_6)
scale1_skip_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_skip_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###############################################
scale1_concat_6 = Concatenate()([scale1_skip_6, scale1_mask_6_up])
scale1_mask_7 = ZeroPadding2D((1,1))(scale1_concat_6)
scale1_mask_7 = Convolution2D(64, (3, 3), activation='relu')(scale1_mask_7)
scale1_mask_7 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_mask_7)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale1_mask_7_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale1_mask_7)


# model1 = Model(inputs = main_input1,outputs = scale1_mask_7_up)
# print(model1.summary())


########################################################################################33
########################################################################################33
##############################scale2 = 0.75###############################################


main_input2 = Input(shape=(240,240,3), dtype='float32', name='main_input2')
scale2_layer_1 = ZeroPadding2D((1,1))(main_input2)
scale2_layer_2 = Convolution2D(64, (3, 3), activation='relu')(scale2_layer_1)
scale2_layer_3 = ZeroPadding2D((1,1))(scale2_layer_2)
scale2_layer_4 = Convolution2D(64, (3, 3), activation='relu')(scale2_layer_3)
scale2_layer_5 = ZeroPadding2D((1,1))(scale2_layer_4)
scale2_pool_1 = MaxPooling2D((3,3), strides=(2,2))(scale2_layer_5)
################ scale2_pool_1 = scale2_layer_5 ##################

scale2_layer_6 = ZeroPadding2D((1,1))(scale2_pool_1)
scale2_layer_7 = Convolution2D(128, (3, 3), activation='relu')(scale2_layer_6)
scale2_layer_8 = ZeroPadding2D((1,1))(scale2_layer_7)
scale2_layer_9 = Convolution2D(128, (3, 3), activation='relu')(scale2_layer_8)
scale2_layer_10 = ZeroPadding2D((1,1))(scale2_layer_9)
scale2_pool_2 = MaxPooling2D((3,3), strides=(2,2))(scale2_layer_10)
################ scale2_pool_2 = scale2_layer_11 ##################

scale2_layer_12 = ZeroPadding2D((1,1))(scale2_pool_2)
scale2_layer_13 = Convolution2D(256, (3, 3), activation='relu')(scale2_layer_12)
scale2_layer_14 = ZeroPadding2D((1,1))(scale2_layer_13)
scale2_layer_15 = Convolution2D(256, (3, 3), activation='relu')(scale2_layer_14)
scale2_layer_16 = ZeroPadding2D((1,1))(scale2_layer_15)
scale2_layer_17 = Convolution2D(256, (3, 3), activation='relu')(scale2_layer_16)
scale2_layer_18 = ZeroPadding2D((1,1))(scale2_layer_17)
scale2_pool_3 = MaxPooling2D((3,3), strides=(2,2))(scale2_layer_18)
################ scale2_pool_3 = scale2_layer_19 ##################

scale2_layer_20 = ZeroPadding2D((1,1))(scale2_pool_3)
scale2_layer_21 = Convolution2D(512, (3, 3), activation='relu')(scale2_layer_20)
scale2_layer_22 = ZeroPadding2D((1,1))(scale2_layer_21)
scale2_layer_23 = Convolution2D(512, (3, 3), activation='relu')(scale2_layer_22)
scale2_layer_24 = ZeroPadding2D((1,1))(scale2_layer_23)
scale2_layer_25 = Convolution2D(512, (3, 3), activation='relu')(scale2_layer_24)
scale2_layer_26 = ZeroPadding2D((1,1))(scale2_layer_25)
scale2_pool_4 = MaxPooling2D((3,3), strides=(1,1))(scale2_layer_26)
################# scale2_pool_4 = scale2_layer_27 ###############

scale2_layer_28 = ZeroPadding2D((2,2))(scale2_pool_4)
scale2_layer_29 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2))(scale2_layer_28)
scale2_layer_30 = ZeroPadding2D((2,2))(scale2_layer_29)
scale2_layer_31 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2))(scale2_layer_30)
scale2_layer_32 = ZeroPadding2D((2,2))(scale2_layer_31)
scale2_layer_33 = Convolution2D(512, (3, 3), activation='relu', dilation_rate=(2,2))(scale2_layer_32)
scale2_layer_34 = ZeroPadding2D((1,1))(scale2_layer_33)
scale2_pool_5 = MaxPooling2D((3,3), strides=(1,1))(scale2_layer_34)
##############scale2_pool_5 = scale2_layer_35################## 

scale2_layer_36 = ZeroPadding2D((12,12))(scale2_pool_5)
scale2_layer_37 = Convolution2D(1024, (3, 3), activation='relu',dilation_rate=(12,12))(scale2_layer_36)
scale2_layer_38 = Dropout(rate = 0.5, noise_shape=None, seed=None)(scale2_layer_37)
scale2_layer_39 = Convolution2D(1024, (1, 1), activation='relu')(scale2_layer_38)
scale2_fc7 = Dropout(rate= 0.5, noise_shape=None, seed=None)(scale2_layer_39)

scale2_mask_1 = Convolution2D(64, (1, 1), strides=(1,1),activation='relu')(scale2_fc7)
scale2_mask_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale2_skip_1 = ZeroPadding2D((1,1))(scale2_fc7)
scale2_skip_1 = Convolution2D(64, (3, 3),activation='relu')(scale2_skip_1)
scale2_skip_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_1)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##############################################################
scale2_concat_1 = Concatenate()([scale2_skip_1, scale2_mask_1])
scale2_concat_1 = ZeroPadding2D((1,1))(scale2_concat_1)
scale2_mask_2 = Convolution2D(64, (3, 3), activation='relu')(scale2_concat_1)
scale2_mask_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale2_skip_2 = ZeroPadding2D((1,1))(scale2_pool_5)
scale2_skip_2 = Convolution2D(64, (3, 3), activation='relu')(scale2_skip_2)
scale2_skip_2 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_2)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###################################################3
scale2_concat_2 = Concatenate()([scale2_skip_2, scale2_mask_2])
scale2_mask_3 = ZeroPadding2D((1,1))(scale2_concat_2)
scale2_mask_3 = Convolution2D(64, (3, 3), activation='relu')(scale2_mask_3)
scale2_mask_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

scale2_skip_3 = ZeroPadding2D((1,1))(scale2_pool_4)
scale2_skip_3 = Convolution2D(64, (3, 3), activation='relu')(scale2_skip_3)
scale2_skip_3 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_3)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

##########################################333
scale2_concat_3 = Concatenate()([scale2_skip_3, scale2_mask_3])
scale2_mask_4 = ZeroPadding2D((1,1))(scale2_concat_3)
scale2_mask_4 = Convolution2D(64, (3, 3), activation='relu')(scale2_mask_4)
scale2_mask_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
# scale2_mask_4_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale2_mask_4)
## upsampling to match dimensions 

scale2_skip_4 = ZeroPadding2D((1,1))(scale2_pool_3)
scale2_skip_4 = Convolution2D(64, (3, 3), activation='relu')(scale2_skip_4)
scale2_skip_4 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_4)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

############################################################
scale2_concat_4 = Concatenate()([scale2_skip_4, scale2_mask_4])
scale2_mask_5 = ZeroPadding2D((1,1))(scale2_concat_4)
scale2_mask_5 = Convolution2D(64, (3, 3), activation='relu')(scale2_mask_5)
scale2_mask_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale2_mask_5_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale2_mask_5)
# scale2_mask_5_up = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale2_mask_5)

scale2_skip_5 = ZeroPadding2D((1,1))(scale2_pool_2)
scale2_skip_5 = Convolution2D(64, (3, 3), activation='relu')(scale2_skip_5)
scale2_skip_5 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_5)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###########################################################
scale2_concat_5 = Concatenate()([scale2_skip_5, scale2_mask_5_up])
scale2_mask_6 = ZeroPadding2D((1,1))(scale2_concat_5)
scale2_mask_6 = Convolution2D(64, (3, 3), activation='relu')(scale2_mask_6)
scale2_mask_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale2_mask_6_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale2_mask_6)

scale2_skip_6 = ZeroPadding2D((1,1))(scale2_pool_1)
scale2_skip_6 = Convolution2D(64, (3, 3), activation='relu')(scale2_skip_6)
scale2_skip_6 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_skip_6)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

###############################################
scale2_concat_6 = Concatenate()([scale2_skip_6, scale2_mask_6_up])
scale2_mask_7 = ZeroPadding2D((1,1))(scale2_concat_6)
scale2_mask_7 = Convolution2D(64, (3, 3), activation='relu')(scale2_mask_7)
scale2_mask_7 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_mask_7)
keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
scale2_mask_7_up = Conv2DTranspose(64,(4,4),strides=2,padding = 'same',activation = 'relu')(scale2_mask_7)


model2 = Model(inputs = main_input2,outputs = scale2_fc7)
print(model2.summary())

########################################################################################33
########################################################################################33
########################################################################################33
########################## Upsampling Ouputs to Same size ###################################


# scale0_output_mask= ZeroPadding2D((1,1))(scale0_mask_7_up)
# scale0_output_mask = Convolution2D(2, (3, 3), activation='relu')(scale0_output_mask)
# scale0_output_mask = Dense(2,kernel_initializer='random_uniform',bias_initializer='zeros')(scale0_output_mask)
# keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

# scale1_output_mask= ZeroPadding2D((1,1))(scale1_mask_7_up)
# scale1_output_mask = Convolution2D(2, (3, 3), activation='relu')(scale1_output_mask)
# scale1_output_mask = Dense(2,kernel_initializer='random_uniform',bias_initializer='zeros')(scale1_output_mask)
# keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

# scale2_output_mask= ZeroPadding2D((1,1))(scale2_mask_7_up)
# scale2_output_mask = Convolution2D(2, (3, 3), activation='relu')(scale2_output_mask),
# scale2_output_mask = Dense(2,kernel_initializer='random_uniform',bias_initializer='zeros')(scale2_output_mask)

# scale1_mask_7_up_interpol = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale1_mask_7_up)

# scale2_mask_7_up_interpol = keras.layers.UpSampling2D(size=(4, 4),interpolation = 'bilinear',data_format=None)(scale2_mask_7_up)
# scale2_mask_7_up_interpol = keras.layers.AveragePooling2D(pool_size=(3, 3),data_format=None)(scale2_mask_7_up_interpol)

# scale1_output_mask_interpol = keras.layers.UpSampling2D(size=(2, 2),interpolation = 'bilinear',data_format=None)(scale1_output_mask)

# scale2_output_mask_interpol = keras.layers.UpSampling2D(size=(4, 4),interpolation = 'bilinear',data_format=None)(scale2_output_mask)
# scale2_output_mask_interpol = keras.layers.AveragePooling2D(pool_size=(3, 3),data_format=None)(scale2_output_mask_interpol)


# ########################################################################################33
# ########################################################################################33
# ########################################################################################33
# ########################## Attention Module ###################################

# mask_7_up_concat = Concatenate()([scale0_mask_7_up, scale1_mask_7_up_interpol, scale2_mask_7_up_interpol])

# att_conv1= ZeroPadding2D((1,1))(mask_7_up_concat)
# att_conv1 = Convolution2D(512, (3, 3), activation='relu')(att_conv1)
# att_conv1 = Dropout(rate = 0.5, noise_shape=None, seed=None)(att_conv1)

# attention = Convolution2D(3, (1, 1), activation='softmax')(att_conv1)

# attention1 = Lambda(lambda attention : attention[:,:,:,0])(attention)
# attention2 = Lambda(lambda attention : attention[:,:,:,1])(attention)
# attention3 = Lambda(lambda attention : attention[:,:,:,2])(attention)

# scale0_output_mask_product = Lambda(dot(attention1,scale0_output_mask))(attention1)


# model2 = Model(inputs = [main_input0,main_input1,main_input2],outputs = scale0_output_mask_product)
# print(model2.summary())

############### ###################











# mask1 = Convolution2D(64, (1, 1), activation='relu')(model)
# mask1_1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(mask1)
# keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

# skip1 = Sequential()Convolution2D(64, (1, 1), strides=(1,1),activation='relu')(fc7)
# mask1 = Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros')(mask1)
# keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
# skip1=ZeroPadding2D((1,1))(model)
# skip1.add(Convolution2D(64, (3, 3), activation='relu'))
# skip1.add(Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros'))
# keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)






	# if weight_path:
	# 	model.load_weights(weight_path)
	# return model


# if __name__ == "__main__":
    # im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    # im[:,:,0] -= 103.939
    # im[:,:,1] -= 116.779
    # im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1))
    # im = np.expand_dims(im, axis=0)
	# get_layer(model_5
