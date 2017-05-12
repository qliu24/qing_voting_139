global category model_category;

object = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
% object = {'aeroplane', 'bicycle', 'motorbike', 'train'};
model_obj = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};

for j = 1:numel(model_obj)
    model_category = model_obj{j}
    for i = 1:numel(object)
        category = object{i}; % set the object of interest
        testVotingModelForBBoxes('config_qing');
    end
end
