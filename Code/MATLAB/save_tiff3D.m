% Written by Lewis Mason

function save_tiff3D(data, path)
    % This function saves a 3D or 2D matrix as a .tif file. Data is a 2D or 3D
    % matrix of values, path is the full path (including name.tif) of the file

    if size(data,3) == 3 % very hardcoded check, could fail, whatever
        % RGB 
        imwrite(data(:,:,:,1),path,'tif',"Compression","none");
        for k = 2:size(data,4)
            imwrite(data(:,:,:,k),path,'tif','WriteMode','append',"Compression","none");
        end
    else
        % Greyscale
        imwrite(data(:,:,1),path,'tif',"Compression","none");
        for k = 2:size(data,3)
            imwrite(data(:,:,k),path,'tif','WriteMode','append',"Compression","none");
        end
    end
end

