% This script masks data for the integer threshold methods for oil and air
% separately

% change these depending on which sample you run
masks_dir = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\Segmented by threshold\1pct\';
full_oil_tif = 'BCTMP pt1pct Fe-R14 Oil_pt1pct Fe-R14 80kV After oil 4501_recon.tif';
full_air_tif = 'Fe0xFe4x1pctHandsheet.tif';


% don't change these
air_dir = 'C:\School\Masters\Scans\Fibre Data\Normalized Fibre Scans\';
oil_dir = 'C:\School\Masters\Scans\Fibre Data\Fibre Scans\Oil\';

all_oil_mask = 'all_oil_mask.tif';
all_air_mask = 'all_air_mask.tif';
just_iron_air_mask = 'just_iron_air_mask.tif';
just_iron_oil_mask = 'just_iron_oil_mask.tif';


% First do in air scans
all_tiff = tiffreadVolume(append(air_dir, full_air_tif));





tmp_mask = tiffreadVolume(append(masks_dir, all_air_mask));
tmp = all_tiff;
tmp(tmp_mask~=1) = 0; % sets to zero for fibre voxels
save_tiff3D(tmp, append(masks_dir, 'all_in_air.tif'));

% tmp = all_tiff;
% tmp(tmp_mask==1) = 0; % sets to zero for fibre voxels
% save_tiff3D(tmp, append(masks_dir, 'bg_in_air.tif'));
% 
% tmp_mask = tiffreadVolume(append(masks_dir, just_iron_air_mask));
% tmp = all_tiff;
% tmp(tmp_mask~=1) = 0; % sets to zero for non tracer voxels
% save_tiff3D(tmp, append(masks_dir, 'tracer_in_air.tif'));
% 
% tmp_mask = tiffreadVolume(append(masks_dir, all_air_mask)) - tiffreadVolume(append(masks_dir, just_iron_air_mask));
% tmp = all_tiff;
% tmp(tmp_mask~=1) = 0; % sets to zero for non tracer UNTRACED fibre voxels
% save_tiff3D(tmp, append(masks_dir, 'fibre_in_air.tif'));



% % Then do in oil
% all_tiff = tiffreadVolume(append(oil_dir, full_oil_tif));
% 
% 
% 
% tmp_mask = tiffreadVolume(append(masks_dir, all_oil_mask));
% tmp = all_tiff;
% tmp(tmp_mask~=1) = 0; % sets to zero for fibre voxels
% save_tiff3D(tmp, append(masks_dir, 'all_in_oil.tif'));
% 
% tmp = all_tiff;
% tmp(tmp_mask==1) = 0; % sets to zero for fibre voxels
% save_tiff3D(tmp, append(masks_dir, 'bg_in_oil.tif'));
% 
% tmp_mask = tiffreadVolume(append(masks_dir, just_iron_oil_mask));
% tmp = all_tiff;
% tmp(tmp_mask~=1) = 0; % sets to zero for non tracer voxels
% save_tiff3D(tmp, append(masks_dir, 'tracer_in_oil.tif'));
% 
% tmp_mask = tiffreadVolume(append(masks_dir, all_oil_mask)) - tiffreadVolume(append(masks_dir, just_iron_oil_mask));
% tmp = all_tiff;
% tmp(tmp_mask~=1) = 0; % sets to zero for non tracer UNTRACED fibre voxels
% save_tiff3D(tmp, append(masks_dir, 'fibre_in_oil.tif'));




