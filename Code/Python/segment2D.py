# This script takes a tiff and runs a pre-determined network over the entire thing. 
# It then outputs (C-1) tiffs, where C is the number of distinct classes and C_0 is air.
# Resulting tiffs are masked input tiff based on predictions by the network
import math
import numpy as np
from tifffile import imsave
from skimage import io # used for importing tif files in a nice way
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from cube_functions import create_square_indices_with_overlap,create_square_indices, create_cube_indices_all, create_squares_from_indices, add_squares_to_data_from_indices
from helper_functions import argmax_and_one_hot
from helper_functions import PlotLearning, custom_accuracy, dice_loss
from tensorflow_addons.optimizers import MultiOptimizer


tiffname = 'Fe3xFibre30kV.tif'
num_classes = 2 # 2 for binarization
cubes_len_yx, num_cubes_zyx, offset_zyx, overlap_yx = [500,500], [2000,100,100], [0,0,0], [0,0]
subset_size = 500 # Determines how many cubes to run through the network at a time, memory limited, different than batch_size

data_dir  = 'C:/School/Masters/Scans/Fibre Data/Normalized Fibre Scans/'
results_dir  = 'C:/School/Masters/Scans/Fibre Data/Normalized Fibre Scans/testing lumen tracking/'

model_dir = 'C:/School/Masters/Thesis/Code/Python/Saved Models/'
model_name = 'tmp2D'
RGB = False

if __name__ == '__main__':
    tiff_full_path = data_dir + tiffname

    print('Loading tiff data')
    data = io.imread(tiff_full_path)

    print("\nUsing model: " + model_name +"\n")

    if False:
        # with overlap currently doesn't work becuase of the z direction?
        indices = create_square_indices_with_overlap(data, cubes_len_yx,num_cubes_zyx, offset_zyx, overlap_yx)
    else:
         indices = create_square_indices(data, cubes_len_yx,num_cubes_zyx, offset_zyx)

    # masks = np.zeros(data.shape[0:-1] + (num_classes,)) # Hardcoded to stop RGB from giving 3 channels

    if RGB == False:
        masks = np.zeros(data.shape + (num_classes,))
    else:
        pass
        # TODO
         

    num_sets    =  math.ceil(len(indices)/subset_size)
    model = tf.keras.models.load_model(model_dir + model_name,
                                custom_objects={"custom_accuracy":custom_accuracy, "dice_loss":dice_loss})

    for i in range(0,num_sets):
        n1 = i * subset_size

        if i == num_sets-1:
            n2 = len(indices)+1
        else:
            n2 = (i+1) * subset_size

        indices_subset = indices[n1:n2]
        print(str(n1) + ' to ' + str(n2) + ' of ' + str(len(indices)))

        cubes = create_squares_from_indices(data, cubes_len_yx, indices_subset)
        results = model.predict(cubes, use_multiprocessing=True, batch_size = 3)
        masks = add_squares_to_data_from_indices(masks, results[:,:,:,:], cubes_len_yx, indices_subset, remove_yx=overlap_yx)
        

    # identical to the one in 3D segment

    print("Creating results directory")
    results_name = tiffname[0:-4] + '_' + model_name + '_'
    dir_name = 'C:/School/Masters/Scans/Fibre Data/Segmented Results/' + results_name
    index = 1   # used for creating unique directory name

    while True:
        results_dir = f"{dir_name}_{index}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            break
        index += 1

    # Argmax and then reconvert to one_hot to maintain shape, extremely inneficient whatever
    print("creating masks")
    masks = argmax_and_one_hot(masks, num_classes, thresh=0.01)


    print("Generating all masks and saving segmented volumes")
    # masks_thresh = 0.01 # Arbitrary
    # masks = masks >= masks_thresh # Make one-hot now, NOTE, I believe this allows 50/50 to be kept, bad?
    for i in range(0,num_classes):
        segmented_path = results_dir + '/'+ tiffname[0:-4] + '_seg_class' + str(i) + '.tif'
        segmented_data = data
        segmented_data = np.multiply(segmented_data, masks[..., i])
        
        num_voxels = np.sum(masks[...,i])
        print('Number of pixels for class ' + str(i) + ':' + str(num_voxels))

        imsave(segmented_path, segmented_data)
        print("Saved file " + segmented_path)

    exit() # no need to do below for now
