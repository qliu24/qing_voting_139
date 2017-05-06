global category;

object = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
% object = {'aeroplane', 'bicycle', 'motorbike', 'train'};
% model_obj = {'bus','bg'};

%  for j = 1:numel(model_obj)
%      model_category = model_obj{j}
    for i = 1:numel(object)
        category = object{i}; % set the object of interest
        testVotingModelForBBoxes2('config_qing');
    end
% end
