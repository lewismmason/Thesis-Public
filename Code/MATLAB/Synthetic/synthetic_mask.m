% This just makes a masked version of a tiff for all classes present, this
% only works for synthetic data with a GT (untested for others)


data_dir = 'C:\School\Masters\Scans\Synthetic Data\Augmentation Example\mixed\';
data_name = 'data.tif';
gt_name = 'ground_truth.tif';

gt = tiffreadVolume(append(data_dir,gt_name));
data = tiffreadVolume(append(data_dir,data_name));


num_classes = max(max(max(gt))); 

for label = 0:num_classes
    masked_data = data; % dumb init to get same size...

    mask = (gt == label);
    
    % Apply mask to the data matrix
    masked_data = data.*uint16(mask);

    save_tiff3D(masked_data, append(data_dir,'data',string(label),data_name));
end
