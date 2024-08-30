# This script generates a 3D volume tiff of fake 2D data. Data can contain squares and circles, and have different intensities for both.
# This also trains the network on the data. Does not currently use cube functionality

# TODO just separate gen_data and train data to different scripts... -> not necessary for my use.

import numpy as np
import cv2 as cv
from tifffile import imsave
import matplotlib.pyplot as plt
import json

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical
import synthetic_network_architectures

# yes this is gross. anyways
import sys
sys.path.insert(1, 'C:\School\Masters\Thesis\Code\Python')
from graphing_functions import save_histogram
from helper_functions import PlotLearning, custom_accuracy, get_opt_tuple_list, hellinger_dist
from network_architectures import IntensityLayer2D

# User specified params
create_new_data = True
train_also      = True


build_architecture = synthetic_network_architectures.basic_convolutional

num_filters = 16
c_layer_init = [4, 20000, 5000] # parameters used to initialize c_layers from a gaussian pdf [num_filters, mean, standard_dev]

# conv_lr = 0.0001
conv_lr = 0.001
c_lr = 100

num_epochs = 50
params_name     = "square and circle"    # This is the tag in the json file for data
# End of user specified params

# Phase params
f = open('synthetic_params.json')
all_params = json.load(f)
params = all_params[params_name]
f.close()

min_radii           = np.array(params["min_radii"])
max_radii           = np.array(params["max_radii"])
phase_means         = np.array(params["phase_means"])
phase_noise_SD      = np.array(params["phase_noise_SD"])
phase_shape_type    = params["phase_shape_type"]            # Circle, Square, maybe others
phase_num_shapes    = np.array(params["phase_num_shapes"])  # Defines the max number of shapes created for each phase (overlap prevention may kill some)
overlap             = params["overlap"]                     # Determines whether or not shapes can overlap

# num_filters         = params["num_filters"] # I never actually use this
batch_size          = params["batch_size"]
# conv_lr       = params["learning_rate_conv"] I removed this, again don't actually use it 
# c_lr       = params["learning_rate_c"]
z_dim, y_dim, x_dim = params["z_dim"],params["y_dim"],params["x_dim"] # doesn't need to be rediculous

num_phases  = len(phase_shape_type) # air is included in this
first_c_thresh = [] # for each class, this contains and integer 2 standard deviations above and below that class's mean, based on that classes standard deviaton (noise), automatically assigned



print("\nHellinger distance of each phase")
for i in range(1, num_phases):
    HD = hellinger_dist(phase_means[i], phase_noise_SD[i], phase_means[i-1],phase_noise_SD[i-1])
    print('Hellinger Distance of class ' + str(i-1) + ' and ' + str(i))
    print(HD)


print('Creating synthetic c-layer thresholds for the first layer, 2 standard deviations above/below each mean')
num_sd_away = 2
for i in range(num_phases):
    if phase_means.ndim == 1: #1D
        first_c_thresh.append(phase_means[i]-phase_noise_SD[i]*num_sd_away)
        first_c_thresh.append(phase_means[i]+phase_noise_SD[i]*num_sd_away)

    elif phase_means.ndim == 2: #3D
        first_c_thresh.append(phase_means[i,:]-phase_noise_SD[i,:]*num_sd_away)
        first_c_thresh.append(phase_means[i,:]+phase_noise_SD[i,:]*num_sd_away)

print('Class means and sd')
print(phase_means)
print(phase_noise_SD)
print('First c-layer thresholds')
print(first_c_thresh)

print('Creating fake data')
# Create 3D volume uint16
if phase_means.ndim == 1:
    num_col_channels = 1
    data        = np.zeros((z_dim, y_dim, x_dim), dtype = np.uint16)
elif phase_means.ndim == 2:
    num_col_channels = 3
    data        = np.zeros((z_dim, y_dim, x_dim, num_col_channels), dtype = np.uint16)

data_labels = np.zeros((z_dim, y_dim, x_dim))

if create_new_data:
    for k in range(0, z_dim):
        if phase_means.ndim == 1:
            tmp = np.ones((y_dim, x_dim), dtype = np.uint16)*phase_means[0]
        elif phase_means.ndim == 2:
            tmp1 = np.ones((y_dim, x_dim), dtype = np.uint16)*phase_means[0][0]
            tmp2 = np.ones((y_dim, x_dim), dtype = np.uint16)*phase_means[0][1]
            tmp3 = np.ones((y_dim, x_dim), dtype = np.uint16)*phase_means[0][2]

        tmp_label = np.zeros((y_dim, x_dim))

        for i in range(1, num_phases):
                num = phase_num_shapes[i]
                origins = np.random.randint(min_radii[i], x_dim-min_radii[i], size=(num, 2)) # note only works since x_dim = y_dim
                radii   = np.random.randint(min_radii[i], max_radii[i], size = num)

                if phase_means.ndim == 1:
                    col     = np.ones(num)*phase_means[i]
                elif phase_means.ndim == 2:
                    # opencv needs BGR
                    col1     = np.ones(num)*phase_means[i][0]
                    col2     = np.ones(num)*phase_means[i][1]
                    col3     = np.ones(num)*phase_means[i][2]

                # Use this for making all phases
                for ii in range(0, num):
                    # if phase_shape_type[i] == 'Circle':
                    if phase_shape_type[i] == 'Circle' and k % 2 == 0 :
                        if phase_means.ndim == 1:
                            cv.circle(tmp, origins[ii,:], radii[ii], col[ii],thickness=-1)
                        elif phase_means.ndim == 2:
                            cv.circle(tmp1, origins[ii,:], radii[ii], col1[ii],thickness=-1)
                            cv.circle(tmp2, origins[ii,:], radii[ii], col2[ii],thickness=-1)
                            cv.circle(tmp3, origins[ii,:], radii[ii], col3[ii],thickness=-1)

                        cv.circle(tmp_label, origins[ii,:], radii[ii], i,thickness=-1) # label

                    # elif phase_shape_type[i] == 'Square':
                    elif phase_shape_type[i] == 'Square'and k % 2 != 0 :
                        if phase_means.ndim == 1:
                            cv.rectangle(tmp, origins[ii,:]-np.array([radii[ii],radii[ii]]), 
                                        origins[ii,:]+np.array([radii[ii],radii[ii]]), col[ii], thickness=-1)
                        elif phase_means.ndim == 2:
                            cv.rectangle(tmp1, origins[ii,:]-np.array([radii[ii],radii[ii]]), 
                                        origins[ii,:]+np.array([radii[ii],radii[ii]]), col1[ii], thickness=-1)
                            cv.rectangle(tmp2, origins[ii,:]-np.array([radii[ii],radii[ii]]), 
                                        origins[ii,:]+np.array([radii[ii],radii[ii]]), col2[ii], thickness=-1)
                            cv.rectangle(tmp3, origins[ii,:]-np.array([radii[ii],radii[ii]]), 
                                        origins[ii,:]+np.array([radii[ii],radii[ii]]), col3[ii], thickness=-1)

                        cv.rectangle(tmp_label, origins[ii,:]-np.array([radii[ii],radii[ii]]), 
                                    origins[ii,:]+np.array([radii[ii],radii[ii]]), i, thickness=-1) # label
                        
                    elif phase_shape_type[i] == 'Rectangle':
                        # Doesn't currently support RGB, i don't really use this anymore so doesnt matter.
                        cv.rectangle(tmp, origins[ii,:]-np.array([radii[ii],radii[ii]-int(radii[ii])]), 
                                    origins[ii,:]+np.array([radii[ii],radii[ii]]), col[ii], thickness=-1)
                        
                        cv.rectangle(tmp_label, origins[ii,:]-np.array([radii[ii],radii[ii]-int(radii[ii])]), 
                                    origins[ii,:]+np.array([radii[ii],radii[ii]]), i, thickness=-1) # label
                        
                    elif phase_shape_type[i] == 'Lumen':
                        # doesn't support RGB, whatever
                        # also the noise isn't applied correctly to the phases, just make it the same for all
                        thickness = 2
                        cv.circle(tmp, origins[ii,:], radii[ii], col[ii],thickness=thickness)
                        # cv.circle(tmp, origins[ii,:], radii[ii]-thickness+1, col[0]-7000,-1)


                        cv.circle(tmp_label, origins[ii,:], radii[ii]-thickness+1, i,thickness=-1) # label



                    else:
                        print('Invalid shape type')

        
        if phase_means.ndim == 1:
            data[k,:,:] = tmp[:,:]
            data_labels[k,:,:] = tmp_label[:,:]
        elif phase_means.ndim == 2:
            data[k,:,:,0] = tmp1[:,:]
            data[k,:,:,1] = tmp2[:,:]
            data[k,:,:,2] = tmp3[:,:]
            data_labels[k,...] = tmp_label[:,:]

    # Add noise yucky code whatever
    for i in range(0,num_phases):
        if phase_means.ndim == 1:
            noise = np.random.normal(0,phase_noise_SD[i], (z_dim,y_dim,x_dim))
            mask = data_labels == i
            data = data + np.multiply(noise,mask)
        elif phase_means.ndim == 2:
            noise1 = np.random.normal(0,phase_noise_SD[i][0], (z_dim,y_dim,x_dim))
            noise2 = np.random.normal(0,phase_noise_SD[i][1], (z_dim,y_dim,x_dim))
            noise3 = np.random.normal(0,phase_noise_SD[i][2], (z_dim,y_dim,x_dim))
            mask = data_labels == i
            
            data[...,0] = data[...,0] + np.multiply(noise1,mask)
            data[...,1] = data[...,1] + np.multiply(noise2,mask)
            data[...,2] = data[...,2] + np.multiply(noise3,mask)


    data = data.astype("uint16") # convert back to correct type


    print('Data shape: ' + str(data.shape) + ', Labels shape: ' + str(data_labels.shape))

    # Save all as tiffs and train
    data_path = 'C:/School/Masters/Scans/Synthetic Data/data.tif'
    label_path = 'C:/School/Masters/Scans/Synthetic Data/ground_truth.tif'
    figure_path = 'C:/School/Masters/Scans/Synthetic Data/histogram_ground_truth' # cant have .png

    if phase_means.ndim == 1:
        save_histogram(data, data_labels, figure_path) # only works for greyscale

    imsave(data_path, data)
    imsave(label_path, data_labels.astype("uint16")) # Save data before we convert to one hot

    print('Synthetic data generated')



if not train_also: exit()



# TODO make it so it runs over squares rather than the full image
data_labels = to_categorical(data_labels, num_classes=num_phases) # change to one-hot encoding, categorical crossentropy

num_outputs = num_phases

 # Create the model, copied from the train code
input_layer = layers.Input((y_dim, x_dim, num_col_channels)) # use zyx because of skimage.io.imread TODO

if True:
    output_layer = build_architecture(input_layer, num_filters, num_outputs, first_c_thresh, c_layer_init) # for EDnet
else:
    output_layer = build_architecture(input_layer, num_filters, num_outputs) # for mot2d pure c layer

model = tf.keras.Model(inputs = input_layer, outputs = output_layer)

# create optimizers, based on the type of layer
conv_opt = tf.keras.optimizers.Adam(learning_rate=conv_lr)
c_opt = tf.keras.optimizers.Adam(learning_rate=c_lr)
optimizers_and_layers = get_opt_tuple_list(model, conv_opt, c_opt)


optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
print(optimizers_and_layers)

model.compile(optimizer = optimizer, 
                loss = 'categorical_crossentropy',
                metrics = ['accuracy', custom_accuracy])

print('\nPrinting initial intensity layer weights (only applicable if an int layer)\n')
print(model.layers[1].get_weights()) 

model.summary()

checkpoint_filepath = 'C:/School/Masters/Thesis/Code/Python/Saved Models/synthetic_tmp'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_custom_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(data, data_labels, 
                batch_size = batch_size, epochs=num_epochs, 
                validation_split = 0.3, shuffle = True,
                use_multiprocessing = True,
                callbacks = [model_checkpoint_callback, PlotLearning()]
                )
