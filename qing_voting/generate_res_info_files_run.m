clear
close all

global category layer_name GPU_id

object = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
config = 'config_voting';
GPU_id = 0;
layer_name = 'pool4';

for i = 1:6
category = object{i};
fprintf('%s\n', category);
fprintf('generate_res_info_files');
generate_res_info_files_bg('train', config);
end

