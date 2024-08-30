% This script crops a target tiff to the desired dimensions, used for very
% large tiffs that my comp cant run over the full thing of. Also squishes
% them as a form of data augmentation for scan consistency

data_name = 'VTT_CTMP_full.tif';


data_dir = 'C:\School\Masters\Scans\Fibre Data\Fibre Scans\';
save_name = 'cropped_and_squished.tif';

des_depth = 900;
des_width = 900;
des_height = 900;

fraction = 1-0.4; 


data = tiffreadVolume(append(data_dir, data_name));
data = data(1:des_height, 1:des_width, 1:des_depth);
data = imresize3(data,fraction);

save_tiff3D(data, append(data_dir, save_name));
