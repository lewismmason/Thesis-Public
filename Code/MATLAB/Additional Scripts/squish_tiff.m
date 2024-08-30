% This script squishes a tiff to a fraction of the original size. This is a
% form of data augmentation

data_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\squished\';
data_name = 'Fe4xFibre30kV.tif';

fraction = 1-0.5;

data = tiffreadVolume(append(data_dir,data_name));
data = imresize3(data,fraction);
save_name = 'squished.tif';

save_tiff3D(data, append(data_dir,save_name));