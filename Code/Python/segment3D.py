# This script takes a tiff and runs a pre-determined network over the entire thing. 
# It then outputs (C) tiffs, where C is the number of distinct classes and C_0 is air.
# Resulting tiffs are masked input tiff based on predictions by the network
import math
import numpy as np
from tifffile import imsave
from skimage import io # used for importing tif files in a nice way
import os

import tensorflow as tf
from cube_functions import create_cube_indices, create_cube_indices_all, create_cubes_from_indices, add_cubes_to_data_from_indices, create_cube_indices_with_overlap
from graphing_functions import save_histogram, get_coloured_accuracy_image
from helper_functions import custom_accuracy, argmax_and_one_hot
from network_architectures import IntensityLayer3D
from tensorflow_addons.optimizers import MultiOptimizer


tiffname = 'Fe4xAndBCTMP_Bundle.tif'
model_name = 'tmp3D'
cubes_len_zyx, num_cubes_zyx, offset_zyx, overlap_zyx = [64,64,64], [2,100,100], [0,0,0], [5,5,5]
# cubes_len_zyx, num_cubes_zyx, offset_zyx, overlap_zyx = [64,64,64], [100,3,100], [0,470,0], [5,5,5]
num_classes = 3
subset_size = 500 # Determines how many cubes to run through the network at a time, memory limited, different than batch_size


data_dir  = 'C:/School/Masters/Scans/Fibre Data/Normalized Fibre Scans/'
data_labels_dir = 'C:/School/Masters/Scans/Fibre Data/Binarized Fibre Scans/' # used for graphically comparing
model_dir = 'C:/School/Masters/Thesis/Code/Python/Saved Models/'



if __name__ == '__main__':
    tiff_full_path = data_dir + tiffname

    print("\nUsing model: " + model_name +"\n")

    print('Loading tiff data')
    data = io.imread(tiff_full_path)

    indices = create_cube_indices_with_overlap(data, cubes_len_zyx,
                                               num_cubes_zyx, offset_zyx, overlap_zyx)
    
    masks = np.zeros(data.shape + (num_classes,))
    num_sets    =  math.ceil(len(indices)/subset_size)
    model = tf.keras.models.load_model(model_dir + model_name,
                                custom_objects={"custom_accuracy":custom_accuracy})

    # Printing model weights for the first layer
    print("Printing network layer 1 weights (k-layer if it exists)")
    print(model.layers[1].get_weights()) 


    for i in range(0,num_sets):
        n1 = i * subset_size

        if i == num_sets-1:
            n2 = len(indices)+1
        else:
            n2 = (i+1) * subset_size

        indices_subset = indices[n1:n2]
        print(str(n1) + ' to ' + str(n2-1) + ' of ' + str(len(indices)))

        cubes = create_cubes_from_indices(data, cubes_len_zyx, indices_subset)
        results = model.predict(cubes, use_multiprocessing=True, batch_size = 3)
        masks = add_cubes_to_data_from_indices(masks, results[:,:,:,:,:], cubes_len_zyx, indices_subset, remove_zyx=overlap_zyx) # The last dimension of masks is the classwise predictions
    
    
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

    # Argmax and then reconvert to one_hot to maintain shape, extremely inneficient whatever ...
    print("creating masks")
    masks = argmax_and_one_hot(masks, num_classes, thresh=0.01)


    print("Generating all masks and saving segmented volumes")
    for i in range(0,num_classes):
        segmented_path = results_dir + '/'+ tiffname[0:-4] + '_seg_class' + str(i) + '.tif'
        segmented_data = data
        segmented_data = np.multiply(segmented_data, masks[..., i])
        
        num_voxels = np.sum(masks[...,i])
        print('Number of pixels for class ' + str(i) + ':' + str(num_voxels))

        imsave(segmented_path, segmented_data)
        print("Saved file " + segmented_path)
