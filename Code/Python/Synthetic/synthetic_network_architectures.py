import tensorflow as tf
from tensorflow.keras import layers

# This allows file from different dir to be imported
import sys
sys.path.insert(1, 'C:\School\Masters\Thesis\Code\Python')
from network_architectures import IntensityLayer2D, IntensityLayer3D


# 2D network architectures



## Blocks


# Block down style M: motivation
def Block2D_downM(input_layer, num_filters):
    cross = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(input_layer)
    return layers.MaxPooling2D((2,2), strides=(2,2))(cross), cross


def Block2D_downMC(input_layer, num_filters, num_int_filters, int_mean, int_SD, initial_weights = [], trainable = True):
    ints = IntensityLayer2D(num_int_filters, int_mean, int_SD, initial_weights, trainable)(input_layer)
    cat = layers.concatenate([input_layer, ints])
    cross = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(cat)
    return layers.MaxPooling2D((2,2), strides=(2,2))(cross), cross


# Block down style 0: standard encoder decoder with 2 convs before maxpool
def Block2D_down0(input_layer, num_filters):
    # returns tuple, (deeper layer, cross (for unet crossing))
    out = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(input_layer)
    cross = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(out)
    return layers.MaxPooling2D((2,2), strides=(2,2))(cross), cross


# Block down style 1: with intensity layer before first conv
def Block2D_down1(input_layer, num_filters, num_int_filters, int_mean, int_SD, initial_weights = [], trainable = True):
    # returns tuple, (deeper layer, cross (for unet crossing))
    ints = IntensityLayer2D(num_int_filters, int_mean, int_SD, initial_weights, trainable)(input_layer)
    cat = layers.concatenate([input_layer, ints])
    conv = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(cat)
    cross = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(conv)
    return layers.MaxPooling2D((2,2), strides=(2,2))(cross), cross

def Block2D_downCompare(input_layer, num_filters1, num_filters2):
    # Same as down0, but we can change the number of filters between the convs
    out = layers.Conv2D(num_filters1, (3,3), activation='relu', padding='same')(input_layer)
    cross = layers.Conv2D(num_filters2, (3,3), activation='relu', padding='same')(out)
    return layers.MaxPooling2D((2,2), strides=(2,2))(cross), cross


# Bottom block style 0: standard UNET
def Block2D_bottomM(input_layer, num_filters):
    return layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(input_layer)


# Bottom block style 0: standard UNET
def Block2D_bottomMC(input_layer, num_filters, num_int_filters, int_mean, int_SD, initial_weights = [], trainable = True):
    ints = IntensityLayer2D(num_int_filters, int_mean, int_SD, initial_weights, trainable)(input_layer)
    cat = layers.concatenate([input_layer, ints])
    return layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(cat)


# Bottom block style 0: standard UNET
def Block2D_bottom0(input_layer, num_filters):
    conv = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(input_layer)
    return layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(conv)


# Bottom block style 1: with intensity layer
def Block2D_bottom1(input_layer, num_filters, num_int_filters, int_mean, int_SD):
    ints = IntensityLayer2D(num_int_filters, int_mean, int_SD)(input_layer)
    cat = layers.concatenate([input_layer, ints])
    conv = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(cat)
    return layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(conv)


# Bottom block style 1: with intensity layer
def Block2D_bottom2(input_layer, num_filters, num_int_filters, int_mean, int_SD):
    ints = IntensityLayer2D(num_int_filters, int_mean, int_SD)(input_layer)
    ints = layers.ReLU()(ints)
    cat = layers.concatenate([input_layer, ints])
    conv = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(cat)
    return layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(conv)


# Motivation block
def Block2D_upM(input_layer_deep, input_layer_shallow, num_filters):
    deep_up = layers.Conv2DTranspose(num_filters, (2,2), strides=(2,2))(input_layer_deep)
    out = layers.concatenate([deep_up, input_layer_shallow])
    out = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(out)
    return out


# Block up style 0: standard UNET
def Block2D_up0(input_layer_deep, input_layer_shallow, num_filters):
    deep_up = layers.Conv2DTranspose(num_filters, (2,2), strides=(2,2))(input_layer_deep)
    out = layers.concatenate([deep_up, input_layer_shallow])
    out = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(out)
    out = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(out)
    return out


## Architectures





## RFL no intensity for initial RFL stuff

# RFL 8
def MOT2D_RFL8(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):
    L1d, L1cross = Block2D_downM(input_layer, num_filters)
    L2 = Block2D_bottomM(L1d, num_filters*2)
    L1u = Block2D_upM(L2, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


# RFL 8
def C_MOT2D_RFL8(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):

    num_int_filters1 = len(first_c_thresh)
    num_int_filters, filter_mean, SD = c_layer_init[0],c_layer_init[1],c_layer_init[2] 

    L1d, L1cross = Block2D_downMC(input_layer, num_filters, num_int_filters1, filter_mean, SD, first_c_thresh)
    L2 = Block2D_bottomMC(L1d, num_filters*2, num_int_filters, filter_mean, SD)
    L1u = Block2D_upM(L2, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)

# RFL 18
def MOT2D_RFL18(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):
    L1d, L1cross = Block2D_downM(input_layer, num_filters)
    L2d, L2cross = Block2D_downM(L1d, num_filters*2)
    L3 = Block2D_bottomM(L2d, num_filters*4)
    L2u = Block2D_upM(L3, L2cross, num_filters*2)
    L1u = Block2D_upM(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


# RFL 18
def C_MOT2D_RFL18(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):

    num_int_filters1 = len(first_c_thresh)
    num_int_filters, filter_mean, SD = c_layer_init[0],c_layer_init[1],c_layer_init[2] 

    L1d, L1cross = Block2D_downMC(input_layer, num_filters, num_int_filters1, filter_mean, SD, first_c_thresh)
    L2d, L2cross = Block2D_downMC(L1d, num_filters*2, num_int_filters, filter_mean, SD)
    L3 = Block2D_bottomMC(L2d, num_filters*4, num_int_filters, filter_mean, SD)
    L2u = Block2D_upM(L3, L2cross, num_filters*2)
    L1u = Block2D_upM(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


#RFL 38
def MOT2D_RFL38(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):
    L1d, L1cross = Block2D_downM(input_layer, num_filters)
    L2d, L2cross = Block2D_downM(L1d, num_filters*2)
    L3d, L3cross = Block2D_downM(L2d, num_filters*4)
    L4 = Block2D_bottomM(L3d, num_filters*8)
    L3u = Block2D_upM(L4, L3cross, num_filters*4)
    L2u = Block2D_upM(L3u, L2cross, num_filters*2)
    L1u = Block2D_upM(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


#RFL 38
def C_MOT2D_RFL38(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):

    num_int_filters1 = len(first_c_thresh)
    num_int_filters, filter_mean, SD = c_layer_init[0],c_layer_init[1],c_layer_init[2] 

    L1d, L1cross = Block2D_downMC(input_layer, num_filters, num_int_filters1, filter_mean, SD, first_c_thresh)
    L2d, L2cross = Block2D_downMC(L1d, num_filters*2, num_int_filters, filter_mean, SD)
    L3d, L3cross = Block2D_downMC(L2d, num_filters*4, num_int_filters, filter_mean, SD)
    L4 = Block2D_bottomMC(L3d, num_filters*8, num_int_filters, filter_mean, SD)
    L3u = Block2D_upM(L4, L3cross, num_filters*4)
    L2u = Block2D_upM(L3u, L2cross, num_filters*2)
    L1u = Block2D_upM(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


# RFL 78
def MOT2D_RFL78(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):
    L1d, L1cross = Block2D_downM(input_layer, num_filters)
    L2d, L2cross = Block2D_downM(L1d, num_filters*2)
    L3d, L3cross = Block2D_downM(L2d, num_filters*4)
    L4d, L4cross = Block2D_downM(L3d, num_filters*8)
    L5 = Block2D_bottomM(L4d, num_filters*16)
    L4u = Block2D_upM(L5, L4cross, num_filters*8)
    L3u = Block2D_upM(L4u, L3cross, num_filters*4)
    L2u = Block2D_upM(L3u, L2cross, num_filters*2)
    L1u = Block2D_upM(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


# RFL with no intensity used to compare with intensity layers


def MUNET2D_RFL14(input_layer, num_filters, num_outputs, ignore):
    L1d, L1cross = Block2D_down0(input_layer, num_filters)
    L2 = Block2D_bottom0(L1d, num_filters*2)
    L1u = Block2D_up0(L2, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def MUNET2D_RFL32(input_layer, num_filters, num_outputs, ignore):
    L1d, L1cross = Block2D_down0(input_layer, num_filters)
    L2d, L2cross = Block2D_down0(L1d, num_filters*2)
    L3 = Block2D_bottom0(L2d, num_filters*4)
    L2u = Block2D_up0(L3, L2cross, num_filters*2)
    L1u = Block2D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)
    

def MUNET2D_RFL68(input_layer, num_filters, num_outputs, ignore):
    L1d, L1cross = Block2D_down0(input_layer, num_filters)
    L2d, L2cross = Block2D_down0(L1d, num_filters*2)
    L3d, L3cross = Block2D_down0(L2d, num_filters*4)
    L4 = Block2D_bottom0(L3d, num_filters*8)
    L3u = Block2D_up0(L4, L3cross, num_filters*4)
    L2u = Block2D_up0(L3u, L2cross, num_filters*2)
    L1u = Block2D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def MUNET2D_RFL140(input_layer, num_filters, num_outputs, ignore):
    L1d, L1cross = Block2D_down0(input_layer, num_filters)
    L2d, L2cross = Block2D_down0(L1d, num_filters*2)
    L3d, L3cross = Block2D_down0(L2d, num_filters*4)
    L4d, L4cross = Block2D_down0(L3d, num_filters*8)
    L5 = Block2D_bottom0(L4d, num_filters*16)
    L4u = Block2D_up0(L5, L4cross, num_filters*8)
    L3u = Block2D_up0(L4u, L3cross, num_filters*4)
    L2u = Block2D_up0(L3u, L2cross, num_filters*2)
    L1u = Block2D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)

def basic_convolutional(input_layer, num_filters, num_outputs, ignore,ignore2):
    output_layer = layers.Conv2D(num_filters, (3,3), activation='relu', padding='same')(input_layer)
    output_layer = layers.Conv2D(num_outputs, (1,1), activation='softmax')(output_layer)
    return output_layer


# Simple pure intensity layer
def intensity2D(input_layer, num_filters, num_outputs, initial_int=[]):
    mean = 20000 # just is the same as normalization for the matlab stuff
    SD = 5000


    int_layer = IntensityLayer2D(4, mean, SD, initial_int,trainable=True)(input_layer)
    output_layer = layers.Conv2D(num_outputs, (1,1), activation='softmax')(int_layer)
    return output_layer


# Combination of receptive feild and intensity, style 1


def MIUNET2D_RFL14(input_layer, num_filters, num_outputs, first_c_thresh = []):
    num_int_filters = 2*num_outputs # 2 for each class

    filter_mean = 30000
    SD = 1000

    L1d, L1cross = Block2D_down1(input_layer, num_filters, num_int_filters, filter_mean, SD, first_c_thresh, True)
    L2 = Block2D_bottom1(L1d, num_filters*2, num_int_filters, filter_mean, SD*2)
    L1u = Block2D_up0(L2, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def MIUNET2D_RFL32(input_layer, num_filters, num_outputs, first_c_thresh = []):
    num_int_filters = 2*num_outputs # 2 for each class
    filter_mean = 24000
    SD = 2000
    
    L1d, L1cross = Block2D_down1(input_layer, num_filters, num_int_filters, filter_mean,SD,first_c_thresh,False)
    L2d, L2cross = Block2D_down1(L1d, num_filters*2, num_int_filters, filter_mean, SD)
    L3 = Block2D_bottom1(L2d, num_filters*4, num_int_filters, filter_mean, SD)
    L2u = Block2D_up0(L3, L2cross, num_filters*2)
    L1u = Block2D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def MIUNET2D_RFL68(input_layer, num_filters, num_outputs, first_c_thresh = []):
    num_int_filters = 2*num_outputs # 2 for each class
    filter_mean = 24000
    SD = 2000


    L1d, L1cross = Block2D_down1(input_layer, num_filters, num_int_filters, filter_mean, SD, first_c_thresh, False)
    L2d, L2cross = Block2D_down1(L1d, num_filters*2, num_int_filters, filter_mean, SD)
    L3d, L3cross = Block2D_down1(L2d, num_filters*4, num_int_filters, filter_mean, SD)
    L4 = Block2D_bottom1(L3d, num_filters*8, num_int_filters, filter_mean, SD)
    L3u = Block2D_up0(L4, L3cross, num_filters*4)
    L2u = Block2D_up0(L3u, L2cross, num_filters*2)
    L1u = Block2D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def MIUNET2D_RFL140(input_layer, num_filters, num_outputs, first_c_thresh = []):
    num_int_filters = 2*num_outputs # 2 for each class
    filter_mean = 24000
    SD = 2000


    L1d, L1cross = Block2D_down1(input_layer, num_filters,num_int_filters, filter_mean, SD, first_c_thresh, False)
    L2d, L2cross = Block2D_down1(L1d, num_filters*2,num_int_filters, filter_mean, SD)
    L3d, L3cross = Block2D_down1(L2d, num_filters*4,num_int_filters, filter_mean, SD)
    L4d, L4cross = Block2D_down1(L3d, num_filters*8,num_int_filters, filter_mean, SD)
    L5 = Block2D_bottom1(L4d, num_filters*16,num_int_filters, filter_mean, SD)
    L4u = Block2D_up0(L5, L4cross, num_filters*8)
    L3u = Block2D_up0(L4u, L3cross, num_filters*4)
    L2u = Block2D_up0(L3u, L2cross, num_filters*2)
    L1u = Block2D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)



# actual ones ive used for stuff, maintains a similar number of total layers between the ones using c-layer and not (not has more filters in some spots)
# first_c_thresh is list of c_layer threshes for the first layer
# c_layer_init [num_int_filters, filter_mean, filter_SD] used to randomize c params with a gaussian (for the deeper layers only)


def EDNET2D_RFL14(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):
    # first_c_thresh and c_layer_init NOT USED, BUT for this specific case they are used to multiply the number of filters at specific parts to make the same number of "layers" for with vs without clayer
    if len(first_c_thresh)!=0: mult1 = len(first_c_thresh)
    else: mult1 = 1

    if len(c_layer_init) == 0: mult2 = 1
    else: mult2 = c_layer_init[0] # this contains num_int_filters

    L1d, L1cross = Block2D_downCompare(input_layer, num_filters,num_filters*mult1)
    L2 = Block2D_bottom0(L1d, num_filters*2)
    L1u = Block2D_up0(L2, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def C_EDNET2D_RFL14(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):
    
    num_int_filters1 = len(first_c_thresh)
    num_int_filters, filter_mean, SD = c_layer_init[0],c_layer_init[1],c_layer_init[2] 

    L1d, L1cross = Block2D_down1(input_layer, num_filters, num_int_filters1, filter_mean, SD, first_c_thresh)
    L2 = Block2D_bottom1(L1d, num_filters*2, num_int_filters, filter_mean, SD)
    L1u = Block2D_up0(L2, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def EDNET2D_RFL32(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):
    # first_c_thresh and c_layer_init NOT USED, BUT for this specific case they are used to multiply the number of filters at specific parts to make the same number of "layers" for with vs without clayer
    
    if len(first_c_thresh)!=0: mult1 = len(first_c_thresh)
    else: mult1 = 1

    if len(c_layer_init) == 0: mult2 = 1
    else: mult2 = c_layer_init[0] # this contains num_int_filters

    L1d, L1cross = Block2D_downCompare(input_layer, num_filters, num_filters*mult1)
    L2d, L2cross = Block2D_downCompare(L1d, num_filters*2, num_filters*2*mult2)
    L3 = Block2D_bottom0(L2d, num_filters*4)
    L2u = Block2D_up0(L3, L2cross, num_filters*2)
    L1u = Block2D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def C_EDNET2D_RFL32(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):

    num_int_filters1 = len(first_c_thresh)
    num_int_filters, filter_mean, SD = c_layer_init[0],c_layer_init[1],c_layer_init[2] 
    
    L1d, L1cross = Block2D_down1(input_layer, num_filters, num_int_filters1, filter_mean, SD, first_c_thresh)
    L2d, L2cross = Block2D_down1(L1d, num_filters*2, num_int_filters, filter_mean, SD)
    L3 = Block2D_bottom1(L2d, num_filters*4, num_int_filters, filter_mean, SD)
    L2u = Block2D_up0(L3, L2cross, num_filters*2)
    L1u = Block2D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def EDNET2D_RFL68(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):
    # first_c_thresh and c_layer_init NOT USED, BUT for this specific case they are used to multiply the number of filters at specific parts to make the same number of "layers" for with vs without clayer
    
    if len(first_c_thresh)!=0: mult1 = len(first_c_thresh)
    else: mult1 = 1

    if len(c_layer_init) == 0: mult2 = 1
    else: mult2 = c_layer_init[0] # this contains num_int_filters
    
    L1d, L1cross = Block2D_down0(input_layer, num_filters*mult1)
    L2d, L2cross = Block2D_down0(L1d, num_filters*2, num_filters*2*mult2)
    L3d, L3cross = Block2D_down0(L2d, num_filters*2, num_filters*4*mult2)
    L4 = Block2D_bottom0(L3d, num_filters*8)
    L3u = Block2D_up0(L4, L3cross, num_filters*4)
    L2u = Block2D_up0(L3u, L2cross, num_filters*2)
    L1u = Block2D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)


def C_EDNET2D_RFL68(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):
    num_int_filters1 = len(first_c_thresh)
    num_int_filters, filter_mean, SD = c_layer_init[0],c_layer_init[1],c_layer_init[2] 
    
    L1d, L1cross = Block2D_down1(input_layer, num_filters, num_int_filters1, filter_mean, SD, first_c_thresh)
    L2d, L2cross = Block2D_down1(L1d, num_filters*2, num_int_filters, filter_mean, SD)
    L3d, L3cross = Block2D_down1(L2d, num_filters*4, num_int_filters, filter_mean, SD)
    L4 = Block2D_bottom1(L3d, num_filters*8, num_int_filters, filter_mean, SD)
    L3u = Block2D_up0(L4, L3cross, num_filters*4)
    L2u = Block2D_up0(L3u, L2cross, num_filters*2)
    L1u = Block2D_up0(L2u, L1cross, num_filters)
    return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)




# def EDNET2D_RFL140(input_layer, num_filters, num_outputs, first_c_thresh = [], c_layer_init = []):
#     # first_c_thresh and c_layer_init NOT USED, BUT for this specific case they are used to multiply the number of filters at specific parts to make the same number of "layers" for with vs without clayer
    
#     if len(first_c_thresh != 0): mult1 = len(first_c_thresh)
#     else: mult1 = 1

#     if len(c_layer_init) == 0: mult2 = 1
#     else: mult2 = c_layer_init[0] # this contains num_int_filters
    
#     L1d, L1cross = Block2D_down0(input_layer, num_filters*mult1)
#     L2d, L2cross = Block2D_down0(L1d, num_filters*2*mult2)
#     L3d, L3cross = Block2D_down0(L2d, num_filters*4*mult2)
#     L4d, L4cross = Block2D_down0(L3d, num_filters*8*mult2)
#     L5 = Block2D_bottom0(L4d, num_filters*16)
#     L4u = Block2D_up0(L5, L4cross, num_filters*8)
#     L3u = Block2D_up0(L4u, L3cross, num_filters*4)
#     L2u = Block2D_up0(L3u, L2cross, num_filters*2)
#     L1u = Block2D_up0(L2u, L1cross, num_filters)
#     return layers.Conv2D(num_outputs, (1,1), activation='softmax', padding='same')(L1u)

