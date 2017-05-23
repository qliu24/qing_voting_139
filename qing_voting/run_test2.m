global category;

object = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
% object = {'aeroplane', 'bicycle', 'motorbike', 'train'};
% model_obj = {'bg4','bg5','bg6'};

% for j = 1:numel(model_obj)
%    model_category = model_obj{j}
    for i = 1:numel(object)
        category = object{i}; % set the object of interest
        testVotingModelForBBoxes('config_qing3');
    end
% end