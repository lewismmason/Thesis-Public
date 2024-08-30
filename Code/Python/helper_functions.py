# This just has additional helper functions and whatnot that don't fit into the cube_functions and graphing functions sections
# Dependencies are disguisting in this project, I care not :)
import keras
from IPython.display import clear_output
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import csv
from network_architectures import IntensityLayer3D, IntensityLayer2D
import time

def get_opt_tuple_list(model, conv_opt, c_opt):
    # This function gives a tuple list with layers and the optimizer for that layer
    # Currently only changes trainable conv layers and trainable c-layers
    # lr can be a schedule or a scalar
    # currently only works for conv and intensity layers

    list = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            list.append((conv_opt, layer))
        if isinstance(layer, tf.keras.layers.Conv3D):
            list.append((conv_opt, layer))

        elif isinstance(layer, IntensityLayer2D):
            list.append((c_opt, layer))
        elif isinstance(layer, IntensityLayer3D):
            list.append((c_opt, layer))
        else:
            pass # do nothing, not a trainable layer
            # layers_lr.append((layer, conv_lr)) # default to the conv learning rate

    return list


# Define singletons, can't be inside functions
tp = tf.keras.metrics.TruePositives()
fp = tf.keras.metrics.FalsePositives()
fn = tf.keras.metrics.FalseNegatives()


# accuracy metric that removes noise class entirely from acc. Also removes true negatives from accuracy metric
# modified dice loss...
def custom_accuracy(y_true, y_pred):
    # This is onlly useful for synthetic data, don't use on real data.
    num_classes = y_true.shape[-1]
    y_pred_one_hot = tf.one_hot(tf.argmax(y_pred, axis = -1),num_classes) # convert prediction to one_hot

    # remove background phase data
    y_pred_one_hot_bg_removed = y_pred_one_hot[..., 1:]
    y_true_bg_removed = y_true[..., 1:]

    # calculate without true negatives, ie acc = TP/(TP + FP + FN). Usually accuracy is (TP+TN)/(TP+TN+FP+FN), but TN will be very large due to background phase being dominant
    # problem, this needs to not be re-created every time

    # TP = tp(y_true_bg_removed, y_pred_one_hot_bg_removed)
    # FP = fp(y_true_bg_removed, y_pred_one_hot_bg_removed)
    # FN = fn(y_true_bg_removed, y_pred_one_hot_bg_removed)

    # accuracy = TP/(TP+FP+FN)

    accuracy = 0.

    # accuracy = accuracy/(num_classes - 1) # take the average of the class based accuracies
    for i in range(0,num_classes-1): # note the sneaky indices shift here
        TP = tp(y_true_bg_removed[...,i], y_pred_one_hot_bg_removed[...,i])
        FP = fp(y_true_bg_removed[...,i], y_pred_one_hot_bg_removed[...,i])
        FN = fn(y_true_bg_removed[...,i], y_pred_one_hot_bg_removed[...,i])

        class_i_acc = TP/(TP+FP+FN)
        # do I need to do this? idk a better way oops. probably slow too, i care not
        tp.reset_state()
        fp.reset_state()
        fn.reset_state()

        accuracy += class_i_acc

    accuracy/=(num_classes-1) # divide by number of classes excluding air

    return accuracy


# TODO is this the correct dice loss? 
def dice_coefficient(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    return (2.0 * intersection + 1e-5) / (union + 1e-5)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


# squared HD
def hellinger_dist(mu_1, sigma_1, mu_2, sigma_2):
    return 1 - np.sqrt(2*sigma_1*sigma_2/(sigma_1**2+sigma_2**2))*np.exp(-1/4*(mu_1-mu_2)**2/(sigma_1**2+sigma_2**2))


class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []


    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=False)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label='train_' + metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()

        # This is old
        # if epoch %10 == 0:
        # plt.show(block=False)
        # plt.pause(1) # wait for a bit to let comp cool down and plot to be viewed, 5 seconds seems fine
        time.sleep(3) # use this to let the comp cool down otherwise it fries...
        
        plt.savefig('training_plot.png')
        plt.close()

        # disgusting way of creating csv data every time. I care not. Prob a much simpler way
        csv_data = [[]]
        csv_metrics = []

        for metric in metrics:
            csv_metrics.append(metric)
            csv_metrics.append('val_'+metric)
        
        csv_data[0] = csv_metrics

        for i in range(len(self.metrics[metrics[0]])):
            row_data = []
            for metric in metrics:
                row_data.append(self.metrics[metric][i])
                row_data.append(self.metrics['val_'+metric][i])

            csv_data.append(row_data)

        # Now save as a csv (overwrite if already exists, since this is safer)
        with open('training_data.csv','w',newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)

        file.close()


# This function takes the predicion vector and simply sets the element with
# highest probability to 1 and the others to 0. Memory intensive, don't have time to fix right now
# This function is absolutely awful for memory. oops :)
def argmax_and_one_hot(masks, num_classes, thresh): 
    one_hot = np.eye(masks.shape[-1], dtype=bool)[np.argmax(masks, axis=-1)]
    return one_hot
    