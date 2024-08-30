% This script generates the histogram of whatever tiff we are
% interested in. Has some options that can be adjusted (flags in the
% function for now...)
data_dir = 'C:\School\Masters\Thesis Actual Data Results For Paper\Figures for thesis\Image Augmentation\';


% Real data name
data_name = 'Fe0xFibre30kV.tif';

% Synthetic name
% data_name = 'data1data.tif';

show_histogram(append(data_dir,data_name));




function show_histogram(path)
    % This function simply displays the intensity PDF of a XuCT scan

    [~,~,ext] = fileparts(path);
    if strcmp(ext,'.tif')
        data = tiffreadVolume(path); 
    elseif strcmp(ext,'.png')
        data = imread(path);
    end

    % Determine number of bins
    num_bins = imfinfo(path).BitDepth;
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
        N(1) = 0;


        % Apply smoothing if needed (only for the post-augmentation scans)
        if false
            N = smooth_histogram(N);
        end


        % logarithmic stuff, make the plot a log normalized plot
        if false
            N_log = log(N+1); % use to plot log
            N_log = N_log/max(N_log); % normalize to 1
        else
            N_log = N;
        end

        % Remove XuCT scan edges from data count
        figure(4)


        % Hardcoded matlab colours (1-4) if doing out-of-order
%         temp_colour = "#0072BD";
%         temp_colour = "#D95319";
%         temp_colour = "#EDB120";
%         temp_colour =  "#7E2F8E"'
%         plot(edges(1:end-1),N_log, 'LineWidth', 1, 'Color', temp_colour); % Used for graphing with a specific colour
        
        
        plot(edges(1:end-1),N_log, 'LineWidth', 1); % Used for graphing
        hold on

        f=gcf;
        exportgraphics(f,'C:\School\Masters\Scans\Fibre Data\Fibre Scans\fig.png','Resolution',600)
    end

    
end




% Smoothing function to average zero bins with the closest non-zero bins,
% written with chatgpt...
function N_smoothed = smooth_histogram(N)
    N_smoothed = N; % Initialize smoothed N
    zero_indices = find(N == 0); % Find indices of zero bins

    for i = zero_indices
        % Find the closest non-zero bins before and after the zero bin
        before = find(N(1:i-1) ~= 0, 1, 'last');
        after = find(N(i+1:end) ~= 0, 1, 'first') + i;

        % If both non-zero bins are found, average them
        if ~isempty(before) && ~isempty(after)
            N_smoothed(i) = (N(before) + N(after)) / 2;
        elseif ~isempty(before) % Only a non-zero bin before
            N_smoothed(i) = N(before);
        elseif ~isempty(after) % Only a non-zero bin after
            N_smoothed(i) = N(after);
        end
    end
end