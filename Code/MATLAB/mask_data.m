% Written by Lewis Mason

% This script requires two "inputs", the data, and a binary mask for the
% data. All it does is mask using the binary mask on the original image
data_filename = 'Fe0xFibre30kV.tif'; % input
label_filename = 'Fe0xFibre30kV.tif'; % mask

% data_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\';
data_dir = 'C:\School\Masters\Scans\Fibre Data\Fibre Scans\';
binary_mask_dir = 'C:\School\Masters\Scans\Fibre Data\Binarized Fibre Scans\';
output_dir = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\Masked data\';

data = tiffreadVolume(append(data_dir, data_filename));  
mask = tiffreadVolume(append(binary_mask_dir, label_filename));
mask = mask>0; % Used if binarization was done where true > 1

% mask = ~mask; % may be required depending on the class to mask
disp(size(data))

if size(data,4) == 3
    % This doesnt actually work because tiffreadvolume can't read a 3D
    % image correctly (only gets first colour channel)
    masked_dataR = data(:,:,:,1).*uint8(mask);
    masked_dataG = data(:,:,:,2).*uint8(mask);
    masked_dataB = data(:,:,:,3).*uint8(mask);
    masked_data = cat(4,masked_dataR,masked_dataG,masked_dataB);
    imshow(masked_data(:,:,1))
    disp(size(masked_data))
else
    disp('here')
    masked_data = data.*uint16(mask);
end

save_tiff3D(masked_data, append(output_dir, append('Masked_',data_filename)))

