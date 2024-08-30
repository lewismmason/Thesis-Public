% This performs image augmentation based on the histogram distributions.
% Fits the largest peak (air) and linearly transforms it to a predetermined
% final value.


data_name = 'Fe0xFibre30kV.tif';
gaussi = 'gauss2'; % This actually must be gauss2 for fe2-4, and gauss3 for fe0-1
desired_mean=10000; 
desired_var = 9.40406E6; % This MUST be greater than the previous variance, ie must increase for good results
num_bins=2^16; % 15000, 2E6 var

% For my fibre data, 20k mean, 2E6 var

% First shift the mean of the data to be consistent between samples
mask_dir = 'C:\School\Masters\Scans\Fibre Data\Binarized Fibre Scans\'; % This is used to generate a mask of the data, we only want to augment based on the air phase
data_dir = 'C:\School\Masters\Scans\Fibre Data\Fibre Scans\'; % The dir that the data is in
shifted_data_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\';  % The dir we save the shifted tiff in


path = append(data_dir,data_name);
mask_path = append(mask_dir, data_name);


% Choose just manual shift or augmentation
if true
    % augmentation
    shifted_data = shift_XuCT_PDF(path, mask_path, desired_mean, num_bins, desired_var,gaussi);
else
    % manual augmentation, created for demonstrational purposes
    data = tiffreadVolume(path);

    val = 2;
    shifted_data = data*val;
    shifted_data = shifted_data-(45776-22787);
end


path = append(shifted_data_dir, data_name);

save_tiff3D(shifted_data, path);



function shifted_data = shift_XuCT_PDF(path, mask_path, desired_mean, num_bins, desired_var, gaussi) 
    % This function shifts the mean of a tiff to a desired spot for some bit
    % resolution. This is to allow consistency between differing datasets

    data = tiffreadVolume(path);

    edges = linspace(0,num_bins-1, num_bins);   % bins might be slightly off
    
    [N, edges] = histcounts(data, edges);
    N(1) = 0;                                   % Remove black XuCT scan edges from data count

    % Fit to a gaussian to get the peak and shift image data to the desired
    % location
    f = fit(edges(1:end-1)', N', gaussi);     % use 2 or 3, just make sure no negative coefficients?for pure uncoated, use 5 for coated or mixed


    switch gaussi
        case 'gauss1'
            a = [f.a1];
            b = [f.b1];
            c = [f.c1];
        case 'gauss2'
            a = [f.a1, f.a2];
            b = [f.b1, f.b2];
            c = [f.c1, f.c2];
        case 'gauss3'
            a = [f.a1, f.a2, f.a3];
            b = [f.b1, f.b2, f.b3];
            c = [f.c1, f.c2, f.c3];
        case 'gauss4'
            a = [f.a1, f.a2, f.a3, f.a4];
            b = [f.b1, f.b2, f.b3, f.b4];
            c = [f.c1, f.c2, f.c3, f.c4];
        case 'gauss5'
            a = [f.a1, f.a2, f.a3, f.a4, f.a5];
            b = [f.b1, f.b2, f.b3, f.b4, f.b5];
            c = [f.c1, f.c2, f.c3, f.c4, f.c5];
    end

    [~, I] = max(a);
    air_mean = b(I);
    air_noise_var = c(I)^2

    % Plot 
    figure(2)
    plot(f, edges(1:end-1), N,'-') % Plots the of the original data, suggested use after show_histogram
    hold on
    grid on


    % Now that we have the things we need, re-get the data (memory
    % limited) and augment it
    data = tiffreadVolume(path);
    
    %%%%% DELETE ME IF NOT NEEDED %%%%%%%
%     data = data(1:900,1:900,1:900); % Change size of data for too large scans


    data = int32(data);% Turn data into signed int32 bit, use 32 so that when converting signed back to unsigned we dont lose data
    data = data - air_mean;% Shift mean to 0

    variance_change = sqrt(desired_var/air_noise_var)


    data = data.*(variance_change); % Perform variance change
    data = data + desired_mean; % Shift to desired spot
    new_0 = (1-air_mean)*variance_change + desired_mean + 10; % add 10 just to make sure we get it % Deal with data that should be 0
    data(data < new_0) = 0; % This removes the black border pixels which effect the histogram

    data = uint16(data); % Turn back to unsigned int16

    % Next re-fit with gaussians just for testing and visual stuff, the
    % actual stuff is done at this point
    [N2, edges] = histcounts(data, edges);
    N2(1) = 0;
    fit2 = 'gauss5'; % generally just use gauss2 to compare air mean/var, but gauss5 to plot nicely
    f2 = fit(edges(1:end-1)', N2', fit2, 'Exclude',N2<5);     % Use two gaussians for good enough accuracy FOR MY APPLICATION
    
    switch fit2
        case 'gauss2'
                a = [f2.a1, f2.a2];
                b = [f2.b1, f2.b2];
                c = [f2.c1, f2.c2];
            case 'gauss3'
                a = [f2.a1, f2.a2, f2.a3];
                b = [f2.b1, f2.b2, f2.b3];
                c = [f2.c1, f2.c2, f2.c3];
            case 'gauss4'
                a = [f2.a1, f2.a2, f2.a3, f2.a4];
                c = [f2.c1, f2.c2, f2.c3, f2.c4];
            case 'gauss5'
                a = [f2.a1, f2.a2, f2.a3, f2.a4, f2.a5];
                c = [f2.c1, f2.c2, f2.c3, f2.c4, f2.c5];
            case 'gauss6'
                a = [f2.a1, f2.a2, f2.a3, f2.a4, f2.a5, f2.a6];
                c = [f2.c1, f2.c2, f2.c3, f2.c4, f2.c5, f2.c6];
    end

    [~, I] = max(a);
    air_noise_var_after_shift = c(I)^2


%     [~, I2] = min(a); % used for delta mu for different number of
%     coatings
%     delta_mu = b(I2)-b(I)
    
    figure(2)
    plot(edges(1:end-1), N2,'-')
%     plot(edges(1:end-1),f2(edges(1:end-1)),'-', 'LineWidth', 4)
    hold on
    grid on
   
    shifted_data = data;
    
end