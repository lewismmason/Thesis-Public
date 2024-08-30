% This script generates the ground truth data for a .tif XuCT scan as well
% as the data inputted into the network
data_name = 'Fe0xFibre30kV.tif';
save_name = 'Binarized.tif';
int_thresh = uint16(26000);
% data = uint16(data < int_thresh); % I use the inverted binarization to be consistent with RUB output


data_dir = 'C:\School\Masters\Scans\Fibre Data\Fibre Scans\'; % The dir that the data is in, always just use un-augmented
path = append(data_dir,data_name);

data = tiffreadVolume(path);

disp("processing")



% data = process_in_2D(data, @integer_thresh); % Set up to have a range of
% binarization functions, but ended up not using them...

data = (data > int_thresh); %  have to do the opposite sometimes imagej


disp("saving");
save_tiff3D(data, append(data_dir, save_name));





% ------------------- Functions
function processed = integer_thresh(data)
    int_thresh = uint16(31250);
    processed = uint16(data <= int_thresh);
end

function testing = test_filter(data)
    % One form of binarization, just for testing right now. These methods
    % Will likely be hardcoded sample by sample
    testing = susan2D(data, 2, 8); % 10 and 5 is pretty nice
%     testing = susan2D(testing, 2, 4); % 10 and 5 is pretty nice
%     testing = susan2D(testing, 2, 4); % 10 and 5 is pretty nice
end

function binarized_data = process_in_2D(data, binarization_method2D)
    % This function loops through a .tif file and will binarize it based on user created function
    % "binarization_method2D". User created function must perform 2D binarization for an image, full
    % 3D is not yet implemented

    binarized_data = zeros(size(data), 'uint16');

%     for k = 1:size(data,3)
    for k = 1:10
        binarized_data(:,:,k) = binarization_method2D(data(:,:,k));
        disp(append(int2str(k),' of ',int2str(size(data,3))));
    end
end


function susan_data2D = susan2D(data2D, int_thresh, dist_thresh) 
    % Warning: This changes uint16 to uint8 temporarily. IE, 16 bit
    % precision is lost, and it becomes 8 bit precise interpolated to 16
    % bit (linear)
    % Perform susan filter on 2D data. int_thresh = -t and dist_thresh = -d.
    uint8_data2D = im2uint8(data2D);

    imwrite(uint8_data2D, 'temp_in.pgm','pgm') % Save as 2D pgm, susan only works on those
    
%     Use SUSAN filter (C code) through command line on 2D pgm
    cmd = append('.\susan.exe temp_in.pgm temp_out.pgm -s -t ', ...
        string(int_thresh), ' -d ', string(dist_thresh));
    status = system(cmd);

    susan_data2D = imread('temp_out.pgm','pgm'); % Load result data
%     susan_data2D = im2uint16(susan_data2D);
    warning('on');
end


