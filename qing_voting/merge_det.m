Data.root_dir2 = '/media/zzs/4TB/qingliu/qing_intermediate/';
dir_det_result = fullfile(Data.root_dir2, 'result');
model_type = 'single';
model_suffix = sprintf('%s.mat', model_type);

dataset_suffix = 'mergelist_rand';
set_type = 'test';
model_category = 'car';

file_det_result_all = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       model_category, 'all', dataset_suffix, set_type, model_suffix));

objects = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
% objects = {'car', 'aeroplane', 'bicycle', 'motorbike'};

det_all = cell(0,1)
for i = 1:numel(objects)
    category = objects{i};
    
    file_det_result = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                   model_category, category, dataset_suffix, set_type, model_suffix));
    
    assert( exist(file_det_result, 'file') > 0 );
    load(file_det_result, 'det');
    for n = 1:length(det)
        det{n}.cat = category;
    end
    det_all = [det_all;det];
end

save(file_det_result_all, 'det_all', '-v7.3');
