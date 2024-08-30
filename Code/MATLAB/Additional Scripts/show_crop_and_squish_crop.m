%This takes a cropped area and squishes it, mimmicking the drying process

scan_name = 'Fe0xFibre30kV.tif';
squished_scan_name = 'SquishedFe0xFibre30kV.tif';

fraction = 1-0.5; % must be the same as was used for the squishing equation

x_start = 270;
y_start = 300;
z_start = 400; 
square_len = 100;


data_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\squished\';
save_dir = data_dir;



figure(1)
data = tiffreadVolume(append(data_dir, scan_name)); 
data = data(y_start:y_start + square_len, x_start:x_start + square_len,z_start);

data = imresize(data, 1000/square_len, 'nearest');
imshow(data);
imwrite(data, append(save_dir,"raw.png"));



x_start = int16(x_start * fraction);
y_start = int16(y_start * fraction);
z_start = int16(z_start * fraction);
square_len = int16(square_len * fraction);


figure(2)
norm= tiffreadVolume(append(data_dir, squished_scan_name)); 
norm = norm(y_start:y_start + square_len, x_start:x_start + square_len,z_start);

norm = imresize(norm, 1000/square_len, 'nearest');
imshow(norm);
imwrite(norm, append(save_dir,"squish.png"));
% imwrite(oil, append(save_dir,"oil.png"));

