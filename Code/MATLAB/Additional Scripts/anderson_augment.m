% this just augments andersons data, very specific to the way he gave to
% me, ignore this...

data_dir = 'C:\School\Masters\Thesis Actual Data Results For Paper\Andersons results\';

full_segmented_name = 'segmented_stack_pt1_oil.tif';
oil_segmented_name = 'iron_segmented_stack_pt1_oil.tif';


iron = tiffreadVolume(append(data_dir,oil_segmented_name));
all = tiffreadVolume(append(data_dir,full_segmented_name));

normal_mask = uint8(all>0) - uint8(iron>0);

normal_fibre = uint16(normal_mask).*uint16(all);
iron = uint16(iron>0).*uint16(all); % applies the mask


save_tiff3D(normal_fibre, append(data_dir,'new_',full_segmented_name))
save_tiff3D(iron, append(data_dir,'new_',oil_segmented_name))