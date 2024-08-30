# This just performs linear regression on a pre-determined set of datapoints.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


x = np.array(range(1,11))
y = 2*x + 3 # Exact solution we pretend not to know

noise = [-0.06771635, -0.47197064,  0.23645562, -1.86972759, \
        -1.01040818,-1.40742633, -0.54597777, -0.73628928, \
        -1.03344551,  0.83280484]

y = y + noise

# Just used chatgpt for this since its just syntax, didn't wanna write it myself

# Create a linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
learning_rate = 0.01
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss='mean_squared_error')

# Custom callback to store trainable parameters
class TrainableParamsCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.weights = []
        self.biases = []
        
    def on_epoch_end(self, epoch, logs=None):
        weights, biases = self.model.layers[0].get_weights()
        self.weights.append(weights[0][0])
        self.biases.append(biases[0])

params_callback = TrainableParamsCallback()

# Train the model
history = model.fit(x, y, epochs=100, callbacks=[params_callback], verbose=0)

# Make predictions
y_pred = model.predict(x)

# Extract the learned parameters
weights, biases = model.layers[0].get_weights()
weight = weights[0][0]
bias = biases[0]

# Print the equation of the line
print(f"The equation of the line is: y = {weight:.4f}x + {bias:.4f}")

# Plot the loss over epochs
plt.figure()
plt.semilogy(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Loss over Epochs (Log Scale)')
plt.savefig('loss_over_epochs_log.png', dpi=600)
plt.close()

# Plot the trainable parameters over epochs
plt.figure()
plt.plot(params_callback.weights, label='w_1')
plt.plot(params_callback.biases, label='w_2')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Trainable Parameters over Epochs')
plt.legend()
plt.savefig('trainable_parameters_over_epochs.png', dpi=600)
plt.close()

# Plot the original data, the predicted line, and the line y=2x+3
plt.figure()
plt.scatter(x, y, marker='o', label='Data Points')  # Original data with 'o' markers
plt.plot(x, y_pred, color='red', label='Predicted Line')  # Predicted line
plt.plot(x, 2*x + 3, 'r--', label='y = 2x + 3')  # Dashed line y = 2x + 3
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data and Predicted Line')
plt.legend()
plt.savefig('original_data_and_predicted_line.png', dpi=600)
plt.close()