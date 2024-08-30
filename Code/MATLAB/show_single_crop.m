%This shows and saves a cropped area of a 3D tiff

%This takes a cropped area and squishes it, mimmicking the drying process
data_dir = 'C:\School\Masters\Thesis Actual Data Results For Paper\Andersons results\After masking\';
scan_name = 'new_iron_segmented_stack_pt1_oil.tif';

greyscale = true;

x_start = 490;
y_start = 88;
z_start = 169;
Dx = 200;
Dy = 100;


% don't touch this stuff now

x_end = x_start + Dx;
y_end = y_start + Dy;


save_dir = data_dir;


figure(1)
if greyscale
    % greyscale
    data = tiffreadVolume(append(data_dir, scan_name));
else
    %rgb to greyscale for a png
    data = imread(append(data_dir, scan_name));
    data = rgb2gray(data);
end
data = data(y_start:y_end, x_start:x_end,z_start);


data = imresize(data, 1000/Dy, 'nearest'); % maker larger and more HD


imshow(data);
imwrite(data, append(data_dir,"raw_cropped.png"));



