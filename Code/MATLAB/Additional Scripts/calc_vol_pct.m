% This script calculates the volume percent/number of voxels for each class
% in result scans. Used to compare methods. Crop values were manually
% determined


data_dir = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\Halves for full imgs\1pctScanBothHalves\';

tracer_name = '1pctTracer.tif';
both_name = 'Masked_Fe0xFe4x1pctHandsheet.tif';
% make sure to use boolean flags below for my data****

tracer_data = tiffreadVolume(append(data_dir, tracer_name));
both_data = tiffreadVolume(append(data_dir, both_name));


if false

    % andersons data is cropped, if using my data crop to same as his
    if false
        % if the 2pct scan

        zstart = 50;
        zend = 50+900;

        xstart = 78;
        xend = xstart + 900;

        ystart = 450;
        yend = ystart + 300;
        
        % 200 x 900 x 900 his
        % 1013 x 989 x 995 mine
        tracer_data = tracer_data(ystart:yend,xstart:xend,zstart:zend);
        both_data = both_data(ystart:yend,xstart:xend,zstart:zend);
    else
        % the 1pct scan
        zstart = 50;
        zend = 50+900;

        xstart = 56;
        xend = xstart + 900;

        ystart = 450;
        yend = ystart + 200;
        
        % 200 x 900 x 900 his
        % 1013 x 989 x 995 mine
        tracer_data = tracer_data(ystart:yend,xstart:xend,zstart:zend);
        both_data = both_data(ystart:yend,xstart:xend,zstart:zend);
    end

else
    % do nothing if his, already cropped
end


num_tracer = nnz(tracer_data);
num_both = nnz(both_data);
num_fibre = num_both-num_tracer;

disp(num_tracer);
disp(num_fibre);
disp(num_tracer/(num_fibre)*100);
