% This script shows the cropped image from a scan (shows original,
% augmented, and masked) based on user params. This is used primarily for
% comparing these cases (for the paper, fig 2)

scan_name = 'Fe0xFe4x2pctHandsheet.tif';

x_start = 210;
y_start = 430;
z_start = 238; 
square_len = 200;

bright = 20000; % Used to brighten the original image (raw)


data_dir = 'C:\School\Masters\Scans\Fibre Data\Fibre Scans\';
norm_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\';
mask_dir = 'C:\School\Masters\Scans\Fibre Data\Binarized Fibre Scans\';


figure(1)
data = tiffreadVolume(append(data_dir, scan_name)); 
data = data(y_start:y_start + square_len, x_start:x_start + square_len,z_start) + bright;
imshow(data);


figure(2)
norm= tiffreadVolume(append(norm_dir, scan_name)); 
norm = norm(y_start:y_start + square_len, x_start:x_start + square_len,z_start);
imshow(norm);


figure(3)
mask = tiffreadVolume(append(mask_dir, scan_name)); 
mask = ~mask(y_start:y_start + square_len, x_start:x_start + square_len,z_start);
imshow(mask);


% figure(4)
% oil = tiffreadVolume(oil_path);
% oil = oil(y_start:y_start + square_len, x_start:x_start + square_len,z_start) + bright;
% imshow(oil);


save_dir = 'C:\School\Masters\Thesis\Code\MATLAB\images\';
data = imresize(data, 1000/square_len, 'nearest');
imwrite(data, append(save_dir,"raw.png"));

norm = imresize(norm, 1000/square_len, 'nearest');
imwrite(norm, append(save_dir,"aug.png"));

mask = imresize(mask, 1000/square_len, 'nearest');
imwrite(mask, append(save_dir,"mask.png"));
% imwrite(oil, append(save_dir,"oil.png"));



