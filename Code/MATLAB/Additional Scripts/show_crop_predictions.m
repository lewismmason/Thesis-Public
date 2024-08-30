% This script shows the cropped results of 3 scans in a directory compared
% to the network input (just pass the results directory)

dir_name = 'Fe0xFe4x2pctHandsheet_tmp3D__1';
data_dir = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\Halves for full imgs\2pctScanBothHalves\';

y_start = 451;
x_start = 190;

Dx = 500;
Dy = 150;

z_start = 248; 


square_len = 100; % old variable now used for scaling

name = append(dir_name(1:strfind(dir_name, '_')-1), '_seg_'); % ugly name stuff to automate
aug_scan_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\';

figure(1)
original = tiffreadVolume(append(aug_scan_dir,dir_name(1:strfind(dir_name, '_')-1),'.tif')); 
original = original(y_start:y_start + Dy, x_start:x_start + Dx,z_start);
imshow(original);

figure(2)
data = tiffreadVolume(append(data_dir, dir_name,'\', name,'class0.tif')); 
data = data(y_start:y_start + Dy, x_start:x_start + Dx,z_start);
imshow(data);


figure(3)
norm= tiffreadVolume(append(data_dir, dir_name,'\', name,'class1.tif')); 
norm = norm(y_start:y_start + Dy, x_start:x_start + Dx,z_start);
imshow(norm);


figure(4)
mask = tiffreadVolume(append(data_dir, dir_name,'\', name,'class2.tif')); 
mask = mask(y_start:y_start + Dy, x_start:x_start + Dx,z_start);
imshow(mask);


save_dir = 'C:\School\Masters\Thesis\Code\MATLAB\images\';

original = imresize(original, 1000/square_len, 'nearest');
imwrite(original, append(save_dir,"pred_original.png"));

data = imresize(data, 1000/square_len, 'nearest');
imwrite(data, append(save_dir,"pred_air.png"));

norm = imresize(norm, 1000/square_len, 'nearest');
imwrite(norm, append(save_dir,"pred_fibre.png"));

mask = imresize(mask, 1000/square_len, 'nearest');
imwrite(mask, append(save_dir,"pred_tracer.png"));

