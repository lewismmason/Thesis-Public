# This script holds various network architectures and layer types.
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# Random normal distribution class for initialization, taken from tensorflow documentation, arbitrary really
class RandomNormal(tf.keras.initializers.Initializer):
  def __init__(self, mean, stddev):
    self.mean = mean
    self.stddev = stddev

  def __call__(self, shape, dtype=None, **kwargs):
    return tf.random.normal(
        shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

  def get_config(self):  # To support serialization
    return {"mean": self.mean, "stddev": self.stddev}


# Custom intensity reference layer for 2D
class IntensityLayer2D(tf.keras.layers.Layer):
    def __init__(self, num_filters, gaussian_mean, gaussian_sd, initial_weights=[], trainable = True):
        # note , if initial_weights != none, uses gaussian, else uses initial weights
        super().__init__()
        self.M = num_filters
        self.u = gaussian_mean
        self.sd = gaussian_sd

        self.initial_weights = initial_weights # initial weights should be numpy array to make it easier
        self.trainable = trainable
        self.initializer = RandomNormal(self.u, self.sd)


    def build(self, input_shape):
        # input shape is (None, Y, X, N)
        self.Y = input_shape[1]
        self.X = input_shape[2]
        self.N = input_shape[3]

        if self.initial_weights != []:
            self.m = self.add_weight(shape=(1, 1, self.M*self.N),
                                        initializer=tf.keras.initializers.Constant(self.initial_weights),
                                        trainable=self.trainable,
                                        name='mean_shift')

        else:
            self.m = self.add_weight(shape=(1, 1, self.M*self.N),
                                        initializer=self.initializer,
                                        trainable=self.trainable,
                                        name='mean_shift')
        
    def call(self, inputs):

        m1 = tf.repeat(self.m, self.Y, axis=0)
        m1 = tf.repeat(m1, self.X, axis=1)

        inputs2 = tf.tile(inputs, [1,1,1,self.M]) # TODO this first index might be troublesome? has questionmark.... TODO 
        return inputs2 - m1
    
# Custom intensity reference layer for 3D
class IntensityLayer3D(tf.keras.layers.Layer):
    def __init__(self, num_filters, gaussian_mean, gaussian_sd, initial_weights=[], trainable = True):
        super().__init__()
        self.M = num_filters
        self.u = gaussian_mean
        self.sd = gaussian_sd

        self.initial_weights = initial_weights # initial weights should be numpy array to make it easier
        self.trainable = trainable

        self.initializer = RandomNormal(self.u, self.sd)

    def build(self, input_shape):
        # input shape is (None, Z, Y, X, N)
        self.Z = input_shape[1]
        self.Y = input_shape[2]
        self.X = input_shape[3]
        self.N = input_shape[4]

        
        if self.initial_weights != []:
            self.m = self.add_weight(shape=(1, 1, 1, self.M*self.N),
                                        initializer=tf.keras.initializers.Constant(self.initial_weights),
                                        trainable=self.trainable,
                                        name='mean_shift')

        else:
            self.m = self.add_weight(shape=(1, 1, 1, self.M*self.N),
                                        initializer=self.initializer,
                                        trainable=self.trainable,
                                        name='mean_shift')

    
    def call(self, inputs):
        m1 = tf.repeat(self.m, self.Z, axis=0)
        m1 = tf.repeat(m1, self.Y, axis=1)
        m1 = tf.repeat(m1, self.X, axis=2)

        inputs2 = tf.tile(inputs, [1,1,1,1,self.M])
        return inputs2 - m1



# Below is a bunch of random architectures


# Binarization layer for fibres, seemed to  work relatively well, but is also rather arbitrary. Simply operates in 2D
def build_binarization2D(input_layer, num_filters):
    # This model was the one that performed a very good binarization at 90% accuracy (Recursive underfitted binarization)
    # Only has 2 outputs
    int_mean = 31000
    int_SD = 1000
    int_layer    = IntensityLayer2D(num_filters,int_mean, int_SD)(input_layer)
    
    conv_base    = layers.Conv2D(num_filters*3, (3,3), activation='relu', padding='same')(int_layer)
    int_layer2   = IntensityLayer2D(2, int_mean, int_SD)(conv_base) # This is troublesome actually, only makes one colour per conv output. 

    cat_layer    = layers.concatenate([int_layer, conv_base, int_layer2])
    output_layer = layers.Conv2D(2, (1,1), activation='softmax')(cat_layer)

    return output_layer


def intensity3D(input_layer, num_filters, num_outputs):
    mean = 17000 # just is the same as normalization for the matlab stuff
    SD = 5000
    int_layer = IntensityLayer3D(num_filters, mean, SD)(input_layer)
    output_layer = layers.Conv3D(num_outputs, (1,1,1), activation='softmax')(int_layer)
    return output_layer


# 2D






# adding in block functionality from synthetic



# Block down style 0: standard UNET
def Block3D_down0(input_layer, num_filters):
    # returns tuple, (deeper layer, cross (for unet crossing))
    out = layers.Conv3D(num_filters, (3,3,3), activation='relu', padding='same')(input_layer)
    cross = layers.Conv3D(num_filters, (3,3,3), activation='relu', padding='same')(out)
    return layers.MaxPooling3D((2,2,2), strides=(2,2,2))(cross), cross


# Block down style 1: intensity layer before convolutions
def Block3D_down1(input_layer, num_filters, num_int_filters, int_mean, int_SD, first_c_thresh = [], trainable = True):
    # returns tuple, (deeper layer, cross (for unet crossing))
    ints = IntensityLayer3D(num_int_filters, int_mean, int_SD,first_c_thresh, trainable)(input_layer)
    cat = layers.concatenate([input_layer, ints])
    conv = layers.Conv3D(num_filters, (3,3,3), activation='relu', padding='same')(cat)
    cross = layers.Conv3D(num_filters, (3,3,3), activation='relu', padding='same')(conv)
    return layers.MaxPooling3D((2,2,2), strides=(2,2,2))(cross), cross


# Bottom block style 0: standard UNET
def Block3D_bottom0(input_layer, num_filters):
    out = layers.Conv3D(num_filters, (3,3,3), activation='relu', padding='same')(input_layer)
    return layers.Conv3D(num_filters, (3,3,3), activation='relu', padding='same')(out)


# Bottom block style 1: with intensity layer
def Block3D_bottom1(input_layer, num_filters, num_int_filters, int_mean, int_SD):
    ints = IntensityLayer3D(num_int_filters, int_mean, int_SD)(input_layer)
    cat = layers.concatenate([input_layer, ints])
    conv = layers.Conv3D(num_filters, (3,3,3), activation='relu', padding='same')(cat)
    return layers.Conv3D(num_filters, (3,3,3), activation='relu', padding='same')(conv)


# Block up style 0: standard UNET
def Block3D_up0(input_layer_deep, input_layer_shallow, num_filters):
    deep_up = layers.Conv3DTranspose(num_filters, (2,2,2), strides=(2,2,2))(input_layer_deep)
    out = layers.concatenate([deep_up, input_layer_shallow])
    out = layers.Conv3D(num_filters, (3,3,3), activation='relu', padding='same')(out)
    out = layers.Conv3D(num_filters, (3,3,3), activation='relu', padding='same')(out)
    return out


# No intensity functionality, just spatial


def MUNET3D_RFL14(input_layer, num_filters, num_outputs):
    # RF = 14, depth of 2, MUNET
    L1d, L1cross = Block3D_down0(input_layer, num_filters)
    L2 = Block3D_bottom0(L1d, num_filters*2)
    L1u = Block3D_up0(L2, L1cross, num_filters)
    return layers.Conv3D(num_outputs, (1,1,1), activation='softmax', padding='same')(L1u)


def MUNET3D_RFL32(input_layer, num_filters, num_outputs):
    L1d, L1cross = Block3D_down0(input_layer, num_filters)
    L2d, L2cross = Block3D_down0(L1d, num_filters*2)
    L3 = Block3D_bottom0(L2d, num_filters*4)
    L2u = Block3D_up0(L3, L2cross, num_filters*2)
    L1u = Block3D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


# adding intensity functionality


def MIUNET3D_RFL14(input_layer, num_filters, num_outputs, first_c_thresh = []):

    
    num_int_filters = 10
    int_mean = 18000 # 30k if using scan1 scan2 scan3
    int_SD = 5000

    L1d, L1cross = Block3D_down1(input_layer, num_filters, num_int_filters, int_mean, int_SD, first_c_thresh,False)
    L2 = Block3D_bottom1(L1d, num_filters*2, num_int_filters, int_mean, int_SD)
    L1u = Block3D_up0(L2, L1cross, num_filters)
    return layers.Conv3D(num_outputs, (1,1,1), activation='softmax', padding='same')(L1u)
    

def MIUNET3D_RFL32(input_layer, num_filters, num_outputs, first_c_thresh = []):
    num_int_filters = 10
    int_mean = 23000
    int_SD = 5000

    L1d, L1cross = Block3D_down1(input_layer, num_filters, num_int_filters, int_mean, int_SD,first_c_thresh,False)
    L2d, L2cross = Block3D_down1(L1d, num_filters*2, num_int_filters, int_mean, int_SD)
    L3 = Block3D_bottom1(L2d, num_filters*4, num_int_filters, int_mean, int_SD)
    L2u = Block3D_up0(L3, L2cross, num_filters*2)
    L1u = Block3D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def MIUNET3D_RFL68(input_layer, num_filters, num_outputs, first_c_thresh = []):
    num_int_filters = 10
    int_mean = 18000
    int_SD = 5000

    L1d, L1cross = Block3D_down1(input_layer, num_filters, num_int_filters, int_mean, int_SD,first_c_thresh,False)
    L2d, L2cross = Block3D_down1(L1d, num_filters*2, num_int_filters, int_mean, int_SD)
    L3d, L3cross = Block3D_down1(L2d, num_filters*4, num_int_filters, int_mean, int_SD)
    L4 = Block3D_bottom1(L3d, num_filters*8, num_int_filters, int_mean, int_SD)
    L3u = Block3D_up0(L4, L3cross, num_filters*4)
    L2u = Block3D_up0(L3u, L2cross, num_filters*2)
    L1u = Block3D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


