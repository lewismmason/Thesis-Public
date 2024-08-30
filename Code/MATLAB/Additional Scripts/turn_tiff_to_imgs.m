% This script converts num_imgs slides in a 3D tiff to png's and stores them
% in the same directory. Only for greyscale right now...

% tiff name, found in this directory
tiff_name = 'Fe0xFibre.tif';
tiff_data_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\';

% Output gets saved in this directory
output_dir_path = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\testing lumen tracking\';


data = tiffreadVolume(append(tiff_data_dir,tiff_name));

% Select number of images to save
num_imgs = 1;
if num_imgs > size(data,3)
    num_imgs = size(data,3);
end

for k = 1:num_imgs
    img = data(:,:,k);
    name = append(output_dir_path, 'img_',string(k),'.png');
    imwrite(img,name)
end
