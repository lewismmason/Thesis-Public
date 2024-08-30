% This script just merges two half tiffs at a common index (both are the
% same size, just merges contents at a split point, ie take data from each)
% Used to resolve memory limitations

merge_point = 535; 

topdatapath = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\Halves for full imgs\SideBySideBothHalves\SideBySide30kV_tmp3D__1\SideBySide30kV_seg_class2.tif';
bottomdatapath = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\Halves for full imgs\SideBySideBothHalves\SideBySide30kV_tmp3D__3\SideBySide30kV_seg_class2.tif';

save_dir = 'C:\School\Masters\Scans\Fibre Data\Segmented Results\Halves for full imgs\SideBySideBothHalves\';
save_name = 'sidebysideTracer.tif';

data1 = uint16(tiffreadVolume(topdatapath)); 
data2 = uint16(tiffreadVolume(bottomdatapath)); 

data1(:,:,merge_point:end) = data2(:,:,merge_point:end);

save_tiff3D(data1, append(save_dir,save_name));