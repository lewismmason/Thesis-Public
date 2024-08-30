% This script takes a directory path, and turns all of the images in it (a
% folder) into a single tiff, maintaining order. Since my network works with tiffs
% already this saves effort. images must all be same sizes etc


dir_path = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\testing lumen tracking\';

tiff_out_name = 'lumen.tif';
tiff_out_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\testing lumen tracking\';


files = dir(dir_path);
file_names = {files.name};
file_names = file_names(~ismember(file_names, {'.', '..'})); % remove . and ..

% use the first file to create the empty dataset
slide_shape = size(imread(append(dir_path, file_names{1})));
num_slides = length(file_names);
data = zeros([slide_shape,num_slides],'uint8');%,'uint8'); % Hardcoded bc the data is silly 8bit in 24bits (the cancer dataset I am using)

for i = 1:num_slides
    current_slide = imread(append(dir_path, file_names{i}));

    if size(current_slide,3) == 3
        data(:,:,:,i) = current_slide(:,:,:);
    else
        data(:,:,i) = current_slide(:,:);
    end
end

save_tiff3D(data, append(tiff_out_dir,tiff_out_name));