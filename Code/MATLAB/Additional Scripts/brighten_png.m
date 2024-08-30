% This just reads a png, brightens all non-black pixels by a scalar

data_dir = 'C:\School\Masters\Thesis Actual Data Results For Paper\Andersons results\After masking\';
data_name = 'iron_air_2pct.png';

data = imread(append(data_dir,data_name));

val = 10000;
data(data>0) = data(data>0) + val;

imwrite(data, append(data_dir,'new.png'))