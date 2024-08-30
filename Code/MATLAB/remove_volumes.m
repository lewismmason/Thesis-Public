% This just removes small volumes from an mostly black image (could be binary output from creating ground truth
% , and saves the result as a binary image, or as a new mask

data_filename = 'Fe0xFibre30kV.tif';
data_dir = 'C:\School\Masters\Thesis Actual Data Results For Paper\Figures for thesis\Thresholding\';

data = tiffreadVolume(append(data_dir, data_filename));

% Currently this just works on the masks, so we have to re-apply them over
% the original image, rip

mask = data > 0; % arbitrary, just > 0
% mask = ~mask;
mask = bwareaopen(mask, 50); % 30-50 seems decent for scan3

% mask = ~mask; % Have to flip for some dumb reason

disp("saving")
save_tiff3D(mask, append(data_dir,'new_',data_filename));
