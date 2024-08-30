% This shows the cropped versions of the integer thresholded images for
% in_air, in_oil, etc. This crops the "all", "bg", "fibre", "iron" for both
% oil, air, ML (=xxxx)
% Hardcoded with file naming scheme

% name = bg_in_xxxx 


in_what = 'air';
dir_name = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\Segmented by threshold\Half and Half 30kV\';



y_start = 320;
x_start = 310;
square_len = 150; % 300 gives a nice resolution
z_start = 359;

name_all = append('all_in_',in_what,'.tif');
name_bg = append('bg_in_',in_what,'.tif');
name_fibre = append('fibre_in_',in_what,'.tif');
name_tracer = append('tracer_in_',in_what,'.tif');

% name = append(dir_name(1:strfind(dir_name, '_')-1), '_seg_'); % ugly name stuff to automate
% data_dir = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\';
% aug_scan_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\';

% figure(1)
all = tiffreadVolume(append(dir_name,name_all)); 
all = all(y_start:y_start + square_len, x_start:x_start + square_len,z_start);
% imshow(all);

% figure(2)
bg = tiffreadVolume(append(dir_name,name_bg)); 
bg = bg(y_start:y_start + square_len, x_start:x_start + square_len,z_start);
% imshow(bg);

% figure(3)
fibre = tiffreadVolume(append(dir_name,name_fibre)); 
fibre = fibre(y_start:y_start + square_len, x_start:x_start + square_len,z_start);
% imshow(fibre);


% figure(4)
tracer = tiffreadVolume(append(dir_name,name_tracer)); 
tracer = tracer(y_start:y_start + square_len, x_start:x_start + square_len,z_start);
% imshow(tracer);


save_dir = append(dir_name, 'cropped\');
imwrite(all, append(save_dir,in_what,"_all.png"));
imwrite(bg, append(save_dir,in_what,"_bg.png"));
imwrite(fibre, append(save_dir,in_what,"_fibre.png"));
imwrite(tracer, append(save_dir,in_what,"_tracer.png"));

