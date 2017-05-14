clear
close all

global category model_category;

object = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
config = 'config_qing2';

for i = 1:6
    category = object{i};
    model_category = object{i};
    fprintf('%s\n', category);
    fprintf('comptFeatForBBoxes\n');
    comptFeatForBBoxes(config);
end
