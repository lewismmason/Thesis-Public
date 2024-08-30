% This script can (optional) be run after running "augment histogram". It
% requires user inputs and linearly transforms the data point by point
% to have the correct histogram ranges. If you don't want a range changed, then just leave it the same
% in x_desired. Note that values at index i-1 must be less than index i
% (ordered)

data_name = 'Fe4xSideBySide.tif';
data_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\';

% MUST BE NON-DESCENDING*****
x_original = [20000, 21000, 24000,32000,44000]; % (from mixed sample) NORMALIZED
x_desired = [20000, 23000, 24000,32000,44000]; % (from training data) NORMALIZED

REALDATA = true;

i_start = 2; % Don't start on the first element since we need i-1
i_end = length(x_original);

if ~REALDATA
    % Tester ones, we use tiffs for the real
    D = linspace(0, 50000,10000); % temporary for now, get tiffs later
    D_orig = D;
else
    D = tiffreadVolume(append(data_dir,data_name));
    D = int32(D); % Convert type to allow negativesr now, the data
    D_orig = D; % This is a copy of the original we use for some reference
end

for i = linspace(i_start, i_end, i_end-1)
    disp(i)
    x_i = x_original(i);
    x_im1 = x_original(i-1);
    xp_i = x_desired(i);
    xp_im1 = x_desired(i-1);

    % D > x_i
    D(D_orig >= x_i) = D(D_orig >= x_i) + (xp_i - x_i - (xp_im1-x_im1)); % Shift outer region first 

    % offsetting
    D = D - xp_im1; % Offset so linear transformation doesn't have addition component
    D_orig = D_orig-x_im1;

    % x_i > D > xp_im1
    indices = D_orig > 0 & D_orig < x_i-x_im1;
    slope = (xp_i-xp_im1)/(x_i-x_im1);
    D(indices) = D(indices) * slope; % Scale linear region to meet outer region new point
 
    % Undo offsets
    D = D + xp_im1; % undo offset
    D_orig = D_orig+x_im1;

end

D = uint16(D); % Convert back to original type

if ~REALDATA
    disp(x_original)
    hold on
    plot(D_orig,D_orig)
    plot(D_orig,D)
else
    save_tiff3D(D, append(data_dir, 'post_augment.tif'));
end