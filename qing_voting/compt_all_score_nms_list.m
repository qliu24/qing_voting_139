Data.root_dir2 = '/media/zzs/4TB/qingliu/qing_intermediate/';
dir_det_result = fullfile(Data.root_dir2, 'VC_file_round1/result');
dataset_suffix = 'mergelist_rand';
layer_name = 'pool4';
category = 'all';
set_type = 'test';

model_type = 'mix'; % or single
model_suffix = sprintf('%s.mat', model_type);


file_det_result_bg1 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'bg1', category, dataset_suffix, set_type, 'single.mat'));
file_det_result_bg2 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'bg2', category, dataset_suffix, set_type, 'single.mat'));
file_det_result_bg3 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'bg3', category, dataset_suffix, set_type, 'single.mat'));
file_det_result_bg4 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'bg4', category, dataset_suffix, set_type, 'single.mat'));
file_det_result_bg5 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'bg5', category, dataset_suffix, set_type, 'single.mat'));
file_det_result_bg6 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'bg6', category, dataset_suffix, set_type, 'single.mat'));
file_det_result_1 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'car', category, dataset_suffix, set_type, model_suffix));
file_det_result_2 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'bus', category, dataset_suffix, set_type, model_suffix));
file_det_result_3 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'aeroplane', category, dataset_suffix, set_type, model_suffix));
file_det_result_4 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'bicycle', category, dataset_suffix, set_type, model_suffix));
file_det_result_5 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'motorbike', category, dataset_suffix, set_type, model_suffix));
file_det_result_6 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'train', category, dataset_suffix, set_type, model_suffix));
file_det_result_all_all = cell(12,1);
file_det_result_all_all{1} = file_det_result_1;
file_det_result_all_all{2} = file_det_result_2;
file_det_result_all_all{3} = file_det_result_3;
file_det_result_all_all{4} = file_det_result_4;
file_det_result_all_all{5} = file_det_result_5;
file_det_result_all_all{6} = file_det_result_6;
file_det_result_all_all{7} = file_det_result_bg1;
file_det_result_all_all{8} = file_det_result_bg2;
file_det_result_all_all{9} = file_det_result_bg3;
file_det_result_all_all{10} = file_det_result_bg4;
file_det_result_all_all{11} = file_det_result_bg5;
file_det_result_all_all{12} = file_det_result_bg6;

img_num_all = 3507;
score_all = cell(img_num_all,1); 
for ff=1:length(file_det_result_all_all)
    assert( exist(file_det_result_all_all{ff}, 'file') > 0 );
    load(file_det_result_all_all{ff}, 'det_all');
    for n = 1: img_num_all
        score_all{n} = [score_all{n} det_all{n}.score];
    end
end

score_rst = cell([img_num_all 1]); 
for n = 1:img_num_all
    for m = 1:6
        score_rst{n} = [score_rst{n} score_all{n}(:,m) - max(score_all{n}(:, 1:end~=m), [], 2)];
    end
end

score_rst2 = cell([img_num_all 1]); 
for n = 1:img_num_all
    for m = 1:6
        score_rst2{n} = [score_rst2{n} score_all{n}(:,m) - max(score_all{n}(:, 7:12), [], 2)];
    end
end

nms_list_all = cell([img_num_all 1]);
for n = 1: img_num_all
    % compute scores for proposal bounding boxes
    num_bbox = size(score_rst{n}, 1);
    boxes{n} = det_all{n}.box;
    assert(num_bbox == size(det_all{n}.score,1));
    
    % do NMS
    score_highest = max(score_rst{n}, [], 2);
    
    % ad hoc thing
%     [~, si] = sort(-score_highest);
%     height = det_all{n}.img_siz(1);
%     width = det_all{n}.img_siz(2);
%     topn=5;
%     bbox_area = zeros(topn,1);
%     for mm = 1:topn
%         bbmm = boxes{n}(si(mm), :);
%         bbmm = [max(ceil(bbmm(1)), 1), max(ceil(bbmm(2)), 1), min(floor(bbmm(3)), width), min(floor(bbmm(4)), height)];
%         bbox_area(mm) = (bbmm(3)-bbmm(1))*(bbmm(4)-bbmm(2));
%     end
%     [~, biggest_i] = max(bbox_area);
%     score_highest(si(biggest_i)) = score_highest(si(biggest_i))+100;
%     
    nms_list_all{n} = nms([boxes{n}, score_highest], 0.3);
    
    if mod(n, 50) == 0
        fprintf(' %d', n);
    end
end 

nms_list_all2 = cell([img_num_all 1]);
for n = 1: img_num_all
    % compute scores for proposal bounding boxes
    boxes{n} = det_all{n}.box;
    
    % do NMS
    score_highest = max(score_rst2{n}, [], 2);
    nms_list_all2{n} = nms([boxes{n}, score_highest], 0.3);
    
    if mod(n, 50) == 0
        fprintf(' %d', n);
    end
end

savefn = fullfile(dir_det_result,'all_score_nms_list.mat');
save(savefn, 'nms_list_all', 'nms_list_all2', 'score_rst', 'score_rst2', '-v7.3');
