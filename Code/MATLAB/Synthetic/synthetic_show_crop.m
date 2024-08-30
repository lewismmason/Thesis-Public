% This script uses a directory and shows the histogram of the data as well
% as cropped images. Used for synthetic data result directories

% This script shows the cropped image from a scan (shows original,
% augmented, and masked) based on user params. This is used primarily for
% comparing these cases (for the paper, fig 2)

dir='C:\School\Masters\Thesis Actual Data Results For Paper\Fig07Greyscale bg and 2 class results\normal\L greater than RFL\DMu_4000_DSig_2000\same as others\';

x_start = 50;
y_start = 1;

z_start = 1; 

line_width = 3;

square_len = 100;

int_threshs = [];
x_lims = [5000, 35000];
y_lims = [0, 1.1];
histogram = true;

% data_dir = 'C:\School\Masters\Scans\Fibre Data\Fibre Scans\';
% norm_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\';
% mask_dir = 'C:\School\Masters\Scans\Fibre Data\Binarized Fibre Scans\';

data_name = 'data.tif';
gt_name = 'ground_truth.tif';
results_name = 'segmented_predictions.tif';

data = tiffreadVolume(append(dir, data_name)); 
gt = tiffreadVolume(append(dir, gt_name)); 
results = tiffreadVolume(append(dir, results_name)); 

% figure(1)
data_crop = data(y_start:y_start + square_len, x_start:x_start + square_len,z_start,:);
data_crop = squeeze(data_crop);
% imshow(data_crop);

% figure(2)
gt_crop = gt(y_start:y_start + square_len, x_start:x_start + square_len,z_start,:);
gt_crop = cast(gt_crop*(256.0/max(max(max(gt)))),'uint8');
gt_crop = squeeze(gt_crop);
% imshow(gt_crop);

% figure(3)
results_crop = results(y_start:y_start + square_len, x_start:x_start + square_len,z_start,:);
results_crop = cast(results_crop*(256.0/max(max(max(gt)))),'uint8');
results_crop = squeeze(results_crop);
% imshow(results_crop);

if histogram
% Create histogram of the classes
data_all = data;
num_classes = max(max(max(gt)));
for i=1:num_classes+1
    % mask data
    if i == 0
        data = data_all;
    else
        data = data_all(gt == i-1);
    end

    % Determine number of bins
    num_bins = imfinfo(append(dir,data_name)).BitDepth;
    num_bins = 2^num_bins;
    
    % Show RGB separately if necessary
    if size(data,ndims(data)) == 3

        num_bins = 2^16; % OVERWRITING BECAUSE THE DATA IS WRONG 

        edges = linspace(0,num_bins-1, num_bins);   % bins might be slightly off
        disp(size(data))
        %R
        [NR, edges] = histcounts(data(:,:,:,1), edges);
        NR(1) = 0;                                   % Remove XuCT scan edges from data count
        figure(1)
        plot(edges(1:end-1),NR);
        hold on; grid on
        title('R channel hist');

        %G
        [NG, edges] = histcounts(data(:,:,:,2), edges);
        NG(1) = 0;                                   % Remove XuCT scan edges from data count
        figure(2)
        plot(edges(1:end-1),NG);
        hold on; grid on
        title('G channel hist');

        %B
        [NB, edges] = histcounts(data(:,:,:,3), edges);
        NB(1) = 0;                                   % Remove XuCT scan edges from data count
        figure(3)
        plot(edges(1:end-1),NB);
        hold on; grid on
        title('B channel hist');
    else
        % Single channel histogram
        edges = linspace(0,num_bins-1, num_bins);   % bins might be slightly off
        [N, edges] = histcounts(data, edges);
        N(1) = 0;                                   % Remove XuCT scan edges from data count
        N = N/max(N); % normalize to have 1 as peak
        figure(1)
        plot(edges(1:end-1),N, 'LineWidth', line_width); % Used for graphing
      
%         plot(edges(1:end-1),N);
        hold on
%         grid on
    end
end

xlim(x_lims)
ylim(y_lims)
if length(int_threshs) > 0
    xline(int_threshs, '--r')
end

end

save_dir = append(dir,'crop\');
mkdir(save_dir)

data_crop = imresize(data_crop, 1000/square_len,"nearest");
gt_crop = imresize(gt_crop, 1000/square_len,"nearest");
results_crop = imresize(results_crop, 1000/square_len,"nearest");

imwrite(data_crop, append(save_dir,"data_crop.png"));
imwrite(gt_crop, append(save_dir,"gt_crop.png"));
imwrite(results_crop, append(save_dir,"results_crop.png"));
saveas(gcf, append(save_dir,'hist.png'));


