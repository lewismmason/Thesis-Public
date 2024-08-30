# This trains the desired architecture (operates in 3D) over a 3D set of data
# Currently this does not go over full images, just a predetermined subset
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
import network_architectures as network_architectures
from cube_functions import create_cube_indices, create_cube_indices_all, create_cubes_from_indices, add_cubes_to_data_from_indices, create_cube_labels_from_indices

from helper_functions import PlotLearning, custom_accuracy,get_opt_tuple_list


from skimage import io # used for importing tif files in a nice way

model_dir = 'C:/School/Masters/Thesis/Code/Python/Saved Models/'
model_name = 'tmp3D'  # if set to tmp3D this just runs the best model from the last training session

# currently only supports 2 tiffs
gt_dir    = 'C:/School/Masters/Scans/Fibre Data/Binarized Fibre Scans/'
data_dir  = 'C:/School/Masters/Scans/Fibre Data/Normalized Fibre Scans/'

tiffname1 = 'Fe0xFibre30kV.tif'
tiffname2 = 'Fe4xFibre30kV.tif'

build_architecture = network_architectures.MIUNET3D_RFL32

cubes_len_zyx, num_cubes_zyx, offset_zyx = [64,64,64], [3,7,7], [0,100,100]

num_filters     = 64 # Remember these are 3D filters, far more computationally expensive
num_outputs     = 3

validation_split= 0.3
conv_lr   = 0.00001
c_lr = 10 # 100 or 10?
num_epochs      = 100
batch_size      = 3

RGB = False # set to true if using RGB data, may be broken since changes have been made

first_c_thresh = [] # if you would like to hand pick values, add all of them here in a list. leave as empty array if not





if __name__ == '__main__':
    print("Loading data from TIFFS")

    print("Training with base learning rate " + str(conv_lr))
    print("Training with c_layer learning rate " + str(c_lr))

    print("Loading dataset 1 from tiff and converting to cubes, both data and labels")
    data_1      = io.imread(data_dir + tiffname1)
    labels_1    = io.imread(gt_dir + tiffname1)
    indices_1       = create_cube_indices(data_1, cubes_len_zyx, num_cubes_zyx, offset_zyx)
    cubes_1         = create_cubes_from_indices(data_1, cubes_len_zyx, indices_1)
    cube_labels_1   = create_cube_labels_from_indices(labels_1, cubes_len_zyx, indices_1, 2)
    
    print("Loading dataset 2 from tiff and converting to cubes, both data and labels")
    data_2      = io.imread(data_dir + tiffname2)
    labels_2    = io.imread(gt_dir + tiffname2)
    indices_2       = create_cube_indices(data_2, cubes_len_zyx, num_cubes_zyx, offset_zyx)
    cubes_2         = create_cubes_from_indices(data_2, cubes_len_zyx, indices_2)
    cube_labels_2   = create_cube_labels_from_indices(labels_2, cubes_len_zyx, indices_2, 1)


    data            = np.append(cubes_1, cubes_2, 0)
    data_labels     = np.append(cube_labels_1, cube_labels_2, 0)

    print("Shape of data = " + str(data.shape))
    print("Shape of data labels = " + str(data_labels.shape))
    print("Cubes created, now creating model and training")

    # Create the model
    if not RGB:
        input_layer = layers.Input((cubes_len_zyx[0], cubes_len_zyx[1], cubes_len_zyx[2], 1)) # NOTE Conv3D requires min_ndim = 5, hence the 1
    else:
        # TODO 
        pass


    output_layer = build_architecture(input_layer, num_filters, num_outputs, first_c_thresh)
    model = tf.keras.Model(inputs = input_layer, outputs = output_layer)

    # optimizer = keras.optimizers.Adam(learning_rate=learning_rate) # Low learning rate is just better, since we technically have so many ground truth's (0.001 good start, 0.0001 good for many filters (30)), 0.1 unstable
    
    # create optimizers, based on the type of layer
    conv_opt = tf.keras.optimizers.Adam(learning_rate=conv_lr)
    c_opt = tf.keras.optimizers.Adam(learning_rate=c_lr)
    optimizers_and_layers = get_opt_tuple_list(model, conv_opt, c_opt)
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    
    
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer = optimizer,
                    loss = loss, # Used when only 1 output, else use categorical cross entropy
                    metrics = ['accuracy', custom_accuracy])
    model.summary()

    # Save the best model as we proceed with training
    checkpoint_filepath = model_dir + model_name
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_custom_accuracy', # val loss or acc
        mode='max',
        save_best_only=True)
    

    history = model.fit(data, data_labels,
                    batch_size = batch_size, epochs=num_epochs,
                    validation_split = validation_split, shuffle = True,
                    use_multiprocessing = True,
                    callbacks = [model_checkpoint_callback, PlotLearning()]
                    )

