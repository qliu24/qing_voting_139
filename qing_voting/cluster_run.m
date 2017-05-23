
clear
close all

global category layer_name GPU_id

object = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
config = 'config_voting';
GPU_id = 0;
layer_name = 'pool4';


fprintf('cluster');
cluster(config);


