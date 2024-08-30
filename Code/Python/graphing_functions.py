import numpy as np
import matplotlib.pyplot as plt


# NOTE DATA_LABELS MUST BE IN ONE-HOT MODE
# used for synthetic data only
def get_coloured_accuracy_image(data, predictions, data_labels = None):
    # This function displays the prediction results overlayed on the original image. If the data_labels exist it can also show what is correct/incorrect
    # predictions and data_labels are the same shape, data_labels is one hot with dimension -1 being the class axis
    max_val = 2**16-1 # 16 bit integer
    alpha = max_val*0.5
    threshold = 10**-2  # arbitrary

    A = predictions
    M = data_labels

    print("printing shapes of stuff")
    print(A.shape)
    print(M.shape)

    colors = {
        0: (0, 0, 0, 0),      # black with no alpha
        1: (max_val, 0, 0, alpha ),    
        2: (0, 0, 0, 0 ),    
        3: (0, max_val, 0, alpha ),
        4: (0, max_val*9/10, 0, alpha ),
        5: (0, max_val, 0, alpha ),
        6: (0, max_val*(85./100.), 0, alpha ),
        7: (max_val, max_val, max_val, alpha ) # white with alpha
    }

    # Initialize the color array to all zeros (black)
    rgba_image = np.zeros((data.shape[0], data.shape[1], data.shape[2], 4), dtype=np.uint16)

    
    # TODO this is too slow right now, use built in numpy functionality
    # Loop over each pixel in the image and assign the appropriate color
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(0,data.shape[2]):
                for l in range(0, A.shape[-1]):

                    # If we don't have ground truth labels, just overlay the colours
                    if M is None:
                        if A[i,j,k,l] > threshold:
                            if l == 0:
                                rgba_image[i,j,k,:] = colors[2]
                            elif l ==1:
                                rgba_image[i,j,k,:] = colors[3]
                            elif l == 2:
                                rgba_image[i,j,k,:] = colors[4]

                    # Else take into account incorrect labels
                    else:
                        if A[i,j,k,l] > threshold and M[i,j,k,l] == 1:
                            # correct prediction, choose colour based on class
                            if l == 0:
                                rgba_image[i,j,k,:] = colors[2] # class 0
                            elif l == 1:
                                rgba_image[i,j,k,:] = colors[3] # class 1
                            elif l == 2:
                                rgba_image[i,j,k,:] = colors[3] # etc
                            elif l == 3:
                                rgba_image[i,j,k,:] = colors[3]
                            elif l == 4:
                                rgba_image[i,j,k,:] = colors[3]
                            elif l == 5:
                                rgba_image[i,j,k,:] = colors[3]
                            elif l == 6:
                                rgba_image[i,j,k,:] = colors[3]

                            # I will never have more classes, stop here

                        # incorrect prediction colour red
                        elif A[i,j,k,l] > threshold and M[i,j,k,l] != 1:
                            rgba_image[i,j,k,:] = colors[1]
                        elif A[i,j,k,l] < threshold and M[i,j,k,l] == 1:
                            rgba_image[i,j,k,:] = colors[1]
                        else:
                            pass


    # convert data to RGB if its not already:
    if data.shape[-1] != 3:
        data = np.repeat(data[:,:,:,np.newaxis],3,axis=3)

    # add alpha channel to RGB image, below done with chatgpt

    alpha = rgba_image[..., 3] / max_val   # Extract the alpha channel from the RGBA image and normalize it to the range [0, 1]
    color_channels = rgba_image[..., :3].astype(np.float64) # Extract the color channels from the RGBA image and cast them to float64
    grayscale_image = data.astype(np.float64) / np.iinfo(np.uint16).max # Normalize the grayscale image to the range [0, 1]

    # Blend the grayscale image with the color channels using the alpha channel as a mask
    blended_image = np.zeros_like(color_channels)
    for z in range(color_channels.shape[0]):
        blended_image[z] = alpha[z][..., np.newaxis] * color_channels[z] + grayscale_image[z][...]

    blended_image = (blended_image * np.iinfo(np.uint16).max).astype(np.uint16) # Convert the blended image back to uint16 and save it
    return blended_image



# NOTE Below are outdated, I use matlab now with the label tiffs to produce much cleaner ones. (for both real and predicted*)



# Show histogram: NOTE MASKS MUST BE IN CATEGORICAL MODE
# TODO make number of figures ie magnification to be a parameter
# TODO add title as a parameter
def save_histogram(data, masks, path):
    max_val = 2**16
    num_bins = 1000
    num_classes = int(np.max(masks) + 1)
    
    # Create the histograms for the different phases
    hist_all, bin = np.histogram(data.ravel(),num_bins,[0,max_val-1])
    hists, edges = masked_histogram(data, masks)

    x_vals = np.arange(0, max_val, max_val/num_bins)

    ymax = np.max(hist_all) # max value will always be found in this since its union of all classes

    plt.plot(x_vals,hist_all)
    for i in range(0,num_classes):
        plt.plot(x_vals, hists[i])
    
    plt.xlim((0,max_val-1))
    plt.title('Histogram of synthetic data')
    plt.xlabel("Intensity bin")
    plt.ylabel("Count of Pixels/Voxels")

    legend_array = ["Full Image"]

    # Hardcoded for a presentation
    legend_array.append("Air")
    legend_array.append("Fibre")
    legend_array.append("Tracer Fibre")
    plt.xlim([10000, 40000])
    


    # for i in range(0,num_classes):
    #     legend_array.append("Class " + str(i))
    plt.legend(legend_array)

    for i in range(1, 200, 50):
        plt.ylim([0,ymax/i])  # set y limits to value for maximum of graph
        plt.savefig(path + str(i))


# This creates a masked histogram, ie only creates histogram of one class
def masked_histogram(matrix, mask):
    max_val = 2**16
    num_bins = 1000

    flat_matrix = matrix.flatten()
    flat_mask = mask.flatten()
    
    unique_masks = np.unique(flat_mask)
    masks = []
    for mask_val in unique_masks:
        mask = (flat_mask == mask_val)
        masks.append(mask)
    
    masked_values = []
    for mask in masks:
        values = flat_matrix[mask]
        masked_values.append(values)
    
    hists = []
    edges = None
    for values in masked_values:
        hist, edges = np.histogram(values,num_bins,[0,max_val-1])
        hists.append(hist)
    
    return hists, edges
