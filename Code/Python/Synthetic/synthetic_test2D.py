# This runs the selected trained network (operating in 2D) over a 2D or 3D tiff
# The output is a nice graphic showing results

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import MultiOptimizer
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

show_results    = True

num_scans = 100

save_dir = 'C:/School/Masters/Thesis/Code/Python/Saved Models/'
model_name = 'synthetic_tmp' 
model = tf.keras.models.load_model(save_dir+model_name, 
                                custom_objects={"custom_accuracy":custom_accuracy})


print("Printing network layer 1 weights (k-layer if it exists)\n")
print(model.layers[1].get_weights()) 
print('\n')

# For fake data
name = 'data.tif'
labels_name = 'ground_truth.tif'
scan_data_path  = 'C:/School/Masters/Scans/Synthetic Data/'
ground_truth_path = scan_data_path + labels_name
data_path = scan_data_path + name
data = io.imread(data_path) # gives zyx
data = data[0:num_scans ,:,:]

data_labels = io.imread(ground_truth_path)
data_labels = data_labels[0:num_scans ,:,:] # just show the first 10

# results = model.evaluate(data,data_labels) # This also prints the results
results = model.predict(data, use_multiprocessing=True)


# argmax the results and save as image, this is just for predicted masks to generate histograms
prediction_labels = np.argmax(results,axis=-1)
save_path = 'C:/School/Masters/Scans/Synthetic Data/' + 'segmented_predictions.tif'
prediction_labels = prediction_labels[:,:,:,np.newaxis]
print("here")
print(prediction_labels.dtype)
imsave(save_path, prediction_labels.astype(np.uint16))



# plot data and results side by side
# Number of samples to display
num_samples = 8

# Randomly select indices to pick samples from the data
indices = np.random.choice(len(data), num_samples, replace=False)
selected_images = data[indices]
selected_predictions = prediction_labels[indices]

# Create a grid of subplots (4 rows, 4 columns)
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for i in range(num_samples):
    row = i // 2
    col = (i % 2) * 2
    
    # Plot the input image
    axes[row, col].imshow(selected_images[i], cmap='gray', vmin=0, vmax=2**16)
    axes[row, col].set_title(f'Input Image {i+1}')
    axes[row, col].axis('off')
    
    # Plot the predicted segmentation
    axes[row, col + 1].imshow(selected_predictions[i], cmap='gray', vmin=0, vmax=2)
    axes[row, col + 1].set_title(f'Predicted Segmentation {i+1}')
    axes[row, col + 1].axis('off')

# Hide any remaining unused subplots
for j in range(4):
    for k in range(4):
        if j * 4 + k >= num_samples * 2:
            axes[j, k].axis('off')

plt.tight_layout()
plt.savefig('data_resutlts.png', dpi=600)
plt.show()



#--------------garb-------------------
# Argmax and then reconvert to one_hot to maintain shape
num_classes=np.max(data_labels)+1
masks = argmax_and_one_hot(results, num_classes=num_classes, thresh=0.01)

print('Result shape is' + str(masks.shape))

data_labels = to_categorical(data_labels, num_classes=num_classes)

# # Create predicted histogram, and overlayed image
# histogram_path = 'C:/School/Masters/Scans/Synthetic Data/histogram_predicted'
# categorical_masks = np.argmax(masks, axis=-1)
# # categorical_masks = results
# save_histogram(data, categorical_masks, histogram_path)

blended_image = get_coloured_accuracy_image(data, masks, data_labels)
save_path = 'C:/School/Masters/Scans/Synthetic Data/' + 'segmented_colour.tif'
imsave(save_path, blended_image)
#--------------garb-------------------




# show data next to predictions:




# now show the weights

# Extract weights of the first convolutional layer
conv_layer = model.layers[1]
weights, biases = conv_layer.get_weights()

# Plot the convolutional kernels as scalar values
n_filters = weights.shape[-1]
n_columns = 4
n_rows = n_filters // n_columns + (n_filters % n_columns > 0)

fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns*2, n_rows*2))

for i in range(n_filters):
    row = i // n_columns
    col = i % n_columns
    ax = axes[row, col]
    kernel = weights[:, :, 0, i]
    
    # Show the kernel values as a heatmap
    cax = ax.matshow(kernel, cmap='viridis')
    
    # Annotate the values
    for (j, k), val in np.ndenumerate(kernel):
        ax.text(k, j, f'{val:.2f}', ha='center', va='center', color='white' if val < (np.max(kernel) - np.min(kernel)) / 2 else 'black')
    
    ax.axis('off')
    ax.set_title(f'Filter {i+1}', pad=0)

plt.tight_layout()

# Save the figure as a PNG file
plt.savefig('convolution_kernels.png', dpi=600)

plt.show()

