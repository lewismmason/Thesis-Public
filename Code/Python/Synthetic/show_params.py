# This script is used to show the intensity parameters in the first layer

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
from tifffile import imsave
from PIL import Image
from keras.utils.np_utils import to_categorical

# This allows file from different dir to be imported
import sys
sys.path.insert(1, 'C:\School\Masters\Thesis\Code\Python')
from graphing_functions import get_coloured_accuracy_image, save_histogram
from helper_functions import PlotLearning, custom_accuracy, argmax_and_one_hot


save_dir = 'C:/School/Masters/Thesis/Code/Python/Saved Models/'
model_name = 'synthetic_tmp'


model = tf.keras.models.load_model(save_dir+model_name, 
                                custom_objects={"custom_accuracy":custom_accuracy})


weights, biases = model.layers[0].get_weights()