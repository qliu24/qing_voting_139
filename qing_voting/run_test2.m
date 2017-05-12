global category model_category;

object = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
% object = {'aeroplane', 'bicycle', 'motorbike', 'train'};
% model_obj = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
model_obj = {'bg1', 'bg2', 'bg3', 'bg4', 'bg5', 'bg6'};
for j = 1:numel(model_obj)
    model_category = model_obj{j}
    for i = 1:numel(object)
        category = object{i}; % set the object of interest
        testVotingModelForBBoxes2('config_qing2');
    end
end
