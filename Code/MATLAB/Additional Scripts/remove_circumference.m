% This script removes the circumfrance from predictions as it is usually
% noisy and filled with garbage.

% User inputs
radius_offset = 15; % The number of circumference pixels to remove, main user variable
tiff_path = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\Halves for full imgs\SideBySideBothHalves\';
tiff_name = 'sidebysidefibre.tif';
left =8;
right = 988;
top = 19;
bottom = 1000;


data = tiffreadVolume(append(tiff_path,tiff_name));

radius = (right-left)/2;

origin = [(top + bottom)/2, (left + right)/2];

[y_count, x_count, z_count] = size(data);
disp(x_count)
disp(y_count)
disp(z_count)


for i = 1:y_count
    for j = 1:x_count
        % Access the element at row i, column j
%         element = matrix(i, j);
        
        val = [i,j];
        len = norm(val-origin);
        if len + radius_offset > radius
            data(i,j,:) = zeros(1,1,z_count);
        end

    end
    disp(i);
end

save_tiff3D(data,append(tiff_path,'border_removed_',tiff_name));

