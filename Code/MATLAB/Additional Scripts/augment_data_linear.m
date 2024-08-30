% This script performs manual augmentation that aligns mean and stretches
% the shape of the data

data_dir = 'C:\School\Masters\Scans\Fibre Data\Fibre Scans\';
data_name = 'Fe0xFibre30kVSquished.tif';

% Target val is int value that you want to stretch to from current
% (current_val)
target_val = 35000;
target_mean = 15000;
current_val = 30000;
current_mean = 22904;

stretch_ratio = (target_val - target_mean)/(current_val-current_mean); 


data = tiffreadVolume(append(data_dir,data_name));

data = int32(data);% Turn data into signed int32 bit, use 32 so that when converting signed back to unsigned we dont lose data
data = data - current_mean;% Shift mean to 0
data = data.*stretch_ratio; % Perform variance change
data = data + target_mean; % Shift to desired spot
new_0 = (1-current_mean)*stretch_ratio + target_mean + 10; % add 10 just to make sure we get it % Deal with data that should be 0
data(data < new_0) = 0;
data = uint16(data);

save_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\';
save_tiff3D(dasyta, append(save_dir,data_name));