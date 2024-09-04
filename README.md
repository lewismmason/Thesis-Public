# Thesis
This directory contains the majority of code related to my thesis. Data is not stored here as it is too large. The work in this repository is entirely written and designed by Lewis Mason. Locations where external sources were are listed (such as forums and ChatGPT)

# Software versions and setup for my laptop 
* x64, NVIDIA RTX 2070 MAX-q (My laptop was used for synthetic data ML, but a different desctop was used for the X-ray scan segmentation paper (An LaVision unit with an RTX4060Ti, required different versions)
* Python 3.10.9
* Compatibilities: 			https://www.tensorflow.org/install/source#gpu
* Tensorflow 2.11.0       		(pip install tensorflow) 
* CUDA 11.2.0 				https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
* Visual studio community x.xx		https://quasar.ugent.be/files/doc/cuda-msvc-compatibility.html
* cuDNN 8.1 ,     			https://developer.nvidia.com/rdp/cudnn-archive
* Must add stuff to path: https://www.youtube.com/watch?v=EmZZsy7Ym-4 good tutorial


# Uses and information
This describes how to use various scripts and the required workflow for the synthetic and real data. Note that many file paths will need to be changed as they were intentionally hardcoded.

## K-Origins
The "K-Origins" layer can be found in Python/network_architectures.py and is currently called "IntensityLayer2D" and "IntensityLayer3D". Examples of how to use it are found in the various network architectures from the same file and in Python/Synthetic/synthetic_network_architectures. Initialization is shown for some of the layers in these architectures with random variables, but manual initialization is shown in Python/train3D.py and Python/Synthetic/gen_data_and_train.py prior to building the models.

## Synthetic test workflow
Note: terminal may have to be opened in the parent directory (Python) and then changed to Synthetic directory (cd Synthetic) so that Python/network_architectures.py exists in path. This is a bug

1. Create the synthetic data parameters using Python/Synthetic/synthetic_params.json which can be either RGB or greyscale
2. Create a network architecture in Python/Synthetic/synthetic_network_architectures.py or use a pre-existing one
3. Create data and train with Python/Synthetic/gen_data_and_train.py based on the "params_name" and "build_architecture". The first part creates the synthetic data and ground truth for it. The second part trains the network
4. Using the trained network, run Python/Synthetic/synthetic_test2D.py to run the network on data. May need to run gen_data_and_train.py again but with the "train_also" flag set to false. Can also do this to test other data for the same network

## Real test workflow
This section outlines how to train and run a model on real data with the pipeline.

### Training

1. Obtain pure X-ray scans of both target classes (only supports two currently)
2. Binarize (low threshold that leaves speckles) with MATLAB/create_gt_labels.m and then remove small volumes and speckles with MATLAB/remove_volumes.m
3. Choose "desired_mean" and "desired_var" which are used in MATLAB/normalize_data.m to normalize each scan, as well as the binarization that was just created. Generally they should be the same for all scans. Make sure the variance is stretched for each scan, or that "desired_var" > var of each scan. You now have two normalized scans, and a ground truth for each
4. Optional: Check the normalization with MATLAB/show_histogram.m or compare it with the raw data. The normalization will look "coloured in"
5. Run Python/train3D.py to train on the scans. It uses a selected "build_architecture" which choses an architecture found in Python/network_architectures.py. This saves a network based on achieving maximum validation scores. Ignore the secondary accuracy metric, it doesn't function correctly.
6. You now have a trained network that can be used for all samples of a specific type (for example, mixed handsheets)

### Testing

1. Obtain raw X-ray scan of test sample
2. Binarize it to just remove the background, not distinguish phases (same as before)
3. Normalize it with the same "desired_mean" and "desired_var" as before, however these actually may need to be "desired_mean + epsilon_1" and "desired_var + epsilon_2", where the epsilons are varied by trial and error, or other knowledge. It is very challening to get these scans to align, generally. And this will take the majority of your time, as you may have to run it through the network to actually see performance
4. Use Python/segment3D.py to run the trained network on the normalized test scan. It will output 3 scans, one containing background, target class 1, and target class 2.
5. To get the hang of it you may need to fiddle around with "desired_mean + epsilon_1" and "desired_var + epsilon_2". If something goes wrong this is likely what did, and also the training "desired_mean" and "desired_var" can be altered independently to see if that helps out.


