% This script counts the number of non-zero pixels in an image

data_name = 'just iron in oil.tif';
data_dir = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\Segmented by threshold\1pct\';

path = append(data_dir,data_name);
data = tiffreadVolume(path);

data = data > 0;
disp(sum(data,"all"))
