# This trains a 2D model over a single tiff volume. If images are not as tiffs, just convert them to using the script
# Currently this only has one class since I just used it for the biomedical stuff. Can change that later
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
import network_architectures as network_architectures
import Synthetic.synthetic_network_architectures as network_architectures2D
from cube_functions import create_square_indices, create_cube_indices_all, create_squares_from_indices, add_squares_to_data_from_indices, create_square_labels_from_indices

from helper_functions import PlotLearning, custom_accuracy, dice_loss,get_opt_tuple_list


from skimage import io # used for importing tif files in a nice way

model_dir = 'C:/School/Masters/Thesis/Code/Python/Saved Models/'
model_name = 'tmp2D'  # if set to tmp3D this just runs the best model from the last training session

# currently only supports 1 tiff
# gt_dir    = 'C:/School/Masters/Scans/Biomedical/EBHI-SEG/Adenocarcinoma/'
# data_dir  = 'C:/School/Masters/Scans/Biomedical/EBHI-SEG/Adenocarcinoma/'
# tiffname1 = 'Adenocarcinoma_image.tif'
# labelname1 = 'Adenocarcinoma_label.tif'


gt_dir    = 'C:/School/Masters/Scans/Fibre Data/Binarized Fibre Scans/'
data_dir  = 'C:/School/Masters/Scans/Fibre Data/Normalized Fibre Scans/'
tiffname1 = 'Fe0xFibre30kV.tif'
labelname1 = 'Fe0xFibre30kV.tif'

build_architecture = network_architectures2D.MIUNET2D_RFL14

# TODO only seems to work with even side lengths?
cubes_len_yx, num_cubes_zyx, offset_zyx = [500,500], [60,5,5], [0,0,0] # cube, square, same thing. I just make them the size of the full images why not

num_filters     = 15 # Remember these are 3D filters, far more computationally expensive
num_outputs     = 2 # simple binary classification

validation_split= 0.3
conv_lr   = 0.0001
c_lr = 100
# learning_rate = 3E-6
num_epochs      = 400
batch_size      = 3

RGB = False # set to true if using RGB data

first_c_thresh = [] # if you would like to hand pick values, add all of them here in a list. leave as empty array if not


if __name__ == '__main__':
    
    print("Training with base learning rate " + str(conv_lr))
    print("Training with c_layer learning rate " + str(c_lr))
    
    print("Loading dataset 1 from tiff and converting to cubes, both data and labels")
    
    data_1      = io.imread(data_dir + tiffname1)
    labels_1    = io.imread(gt_dir + tiffname1)

    print(data_1.shape)

    indices_1       = create_square_indices(data_1, cubes_len_yx, num_cubes_zyx, offset_zyx)
    cubes_1         = create_squares_from_indices(data_1, cubes_len_yx, indices_1)
    cube_labels_1   = create_square_labels_from_indices(labels_1, cubes_len_yx, indices_1, False)
    
    print(data_1.shape)
    print(labels_1.shape)


    print("Shape of data = " + str(data_1.shape))
    print("Shape of data labels = " + str(labels_1.shape))
    print("Cubes created, now creating model and training")

    # Create the model
    # TODO make RGB checker
    if not RGB:
        input_layer = layers.Input((cubes_len_yx[0], cubes_len_yx[1], 1)) # if RGB this should be a 3 in the final index
    else:
        #TODO
        pass


    output_layer = build_architecture(input_layer, num_filters, num_outputs, first_c_thresh)
    model = tf.keras.Model(inputs = input_layer, outputs = output_layer)

    # optimizer = keras.optimizers.Adam(learning_rate=learning_rate) # Low learning rate is just better, since we technically have so many ground truth's (0.001 good start, 0.0001 good for many filters (30)), 0.1 unstable
    
    # create optimizers, based on the type of layer
    conv_opt = tf.keras.optimizers.Adam(learning_rate=conv_lr)
    c_opt = tf.keras.optimizers.Adam(learning_rate=c_lr)
    optimizers_and_layers = get_opt_tuple_list(model, conv_opt, c_opt)
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    
    
    loss      = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer = optimizer,
                    loss = loss, # Used when only 1 output, else use categorical cross entropy
                    metrics = ['accuracy', custom_accuracy, dice_loss])
    model.summary()

    # Save the best model as we proceed with training
    checkpoint_filepath = model_dir + model_name
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy', # val loss or acc
        mode='max',
        save_best_only=True)
    


    history = model.fit(cubes_1, cube_labels_1,
                    batch_size = batch_size, epochs=num_epochs,
                    validation_split = validation_split, shuffle = True,
                    use_multiprocessing = True,
                    callbacks = [model_checkpoint_callback, PlotLearning()]
                    )


# Save the figure afterwards
plt.figure(figsize=(10, 5))
plt.plot(history.history['custom_accuracy'])
plt.plot(history.history['val_custom_accuracy'])
plt.title('Model Custom Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig('accuracy_plot.png')
