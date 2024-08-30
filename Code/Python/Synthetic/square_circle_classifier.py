# Traditional convolutional classifier to predict if an image has a square or a circle in it, not semantic
# This is basically a copy past of the code in #gen_data_and_train, but I don't want to break that code.
# I just used chatgpt for the architecture stuff and plotting etc.

# IDK what I actually need here, doesn't matter tbh

import numpy as np
import cv2 as cv
from tifffile import imsave
import matplotlib.pyplot as plt
import json

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from keras.utils.np_utils import to_categorical
import synthetic_network_architectures
from sklearn.model_selection import train_test_split

from skimage import io
import sys
sys.path.insert(1, 'C:\School\Masters\Thesis\Code\Python')
from graphing_functions import save_histogram
from helper_functions import PlotLearning, custom_accuracy, get_opt_tuple_list, hellinger_dist
from network_architectures import IntensityLayer2D






# For fake data, reqwuries it to be already created from "gen_data_and_train"
name = 'data.tif'
scan_data_path  = 'C:/School/Masters/Scans/Synthetic Data/'
data_path = scan_data_path + name
data = io.imread(data_path) # gives zyx


#chatgpt for the rest, saves me time


sequence = np.array([1, 0])
labels = np.tile(sequence, 500)

# Reshape data to add a channel dimension (z, width, height, channels)
data = data.reshape((1000, 7, 7, 1))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# Define the model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(7, 7, 1)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

# history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32,
#                     callbacks = [PlotLearning()])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Make predictions on the test set
predictions = model.predict(X_test)
# Convert probabilities to binary labels
predicted_labels = (predictions > 0.5).astype(int)

# Print some predictions
print(f"Predictions: {predicted_labels[:10].flatten()}")
print(f"True Labels: {y_test[:10]}")


# show the data
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(7, 7), cmap='gray', vmin=0, vmax=2**16)
    ax.set_title(f'Pred: {predicted_labels[i][0]}, True: {y_test[i]}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('data_results.png', dpi=600)
plt.show()





# Extract weights of the first convolutional layer
conv_layer = model.layers[0]
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