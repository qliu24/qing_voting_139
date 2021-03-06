function det_debug()
config='config_qing3';
try
    eval(config)
catch
    keyboard
end
%%
fprintf('Evaluate voting models for object detection task on "%s" set ...\n', set_type);

% read image list
img_list_all = cell(1, 3);
objects = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
for oi = 1:numel(objects)
    oo = objects{oi};
    switch set_type
        case 'train'
            file_list = sprintf(Dataset.train_list, oo);
        case 'test'
            file_list = sprintf(Dataset.test_list, oo);
        otherwise
            error('Error: unknown set_type');
    end
    assert(exist(file_list, 'file') > 0);
    file_ids = fopen(file_list, 'r');
    img_list_i = textscan(file_ids, '%s %d');
    img_list_all{1} = [img_list_all{1}; img_list_i{1}];
    img_list_all{2} = [img_list_all{2}; img_list_i{2}];
    oolist = cell(length(img_list_i{1}), 1);
    oolist(:) = {oo};
    img_list_all{3} = [img_list_all{3}; oolist];
    
    if strcmp(oo, model_category)
        img_list = img_list_i;
        img_num = length(img_list{1});
    end
end

dir_img = sprintf(Dataset.img_dir, model_category);
dir_obj_anno = sprintf(Dataset.anno_dir, model_category);

                                                 
%% process ground-truth annotations
load(file_gt_obj_anno, 'gt', 'n_pos');

fprintf(' compute scores and do NMS for detection result:');

% load detection result
assert( exist(file_det_result_all, 'file') > 0 );
load(file_det_result_all, 'det_all');
det_all_m = det_all;

all_rst_file = fullfile(dir_det_result,'all_score_nms_list.mat');
assert( exist(all_rst_file, 'file') > 0 );
load(all_rst_file);

% assert(length(det_all) == img_num);
img_num_all = length(nms_list_all);
assert(length(gt) == img_num);
assert(length(img_list_all{1}) == img_num_all);

boxes = cell([img_num_all 1]);              % ~ boxes{n}: ~ [num_bbox 4]
scores = cell([img_num_all 1]);             % ~ scores{n}: ~ [num_bbox 1]
% index for gt
img_ids = cell([img_num_all 1]);            % ~ img_ids{n}: ~ [num_bbox 1]
% index for img_list
img_ids_all = cell([img_num_all 1]);   

obj_col = containers.Map;
obj_col('car') = 1;
obj_col('bus') = 2;
obj_col('aeroplane') = 3;
obj_col('bicycle') = 4;
obj_col('motorbike') = 5;
obj_col('train') = 6;

n_obj = 0;
for n = 1: img_num_all
    % compute scores for proposal bounding boxes
    num_bbox = size(det_all_m{n}.score, 1);
    scores{n} = score_rst{n}(:,obj_col(model_category));
    boxes{n} = det_all_m{n}.box;
    
    nms_list = nms_list_all2{n};
    % nms_list = nms([boxes{n}, scores{n}], Eval.nms_bbox_ratio);
    boxes{n} = boxes{n}(nms_list, :);
    scores{n} = scores{n}(nms_list);
    
    img_ids_all{n} = n * ones([length(nms_list) 1]);
    
    if strcmp(det_all_m{n}.cat, model_category)
        n_obj = n_obj+1;
        img_ids{n} = n_obj * ones([length(nms_list) 1]);
    else
        img_ids{n} = 0 * ones([length(nms_list) 1]);
    end
    
%     valid_score_box = find(scores{n} > Eval.score_thresh);
%     if ~isempty(valid_score_box)
%         scores{n} = scores{n}(valid_score_box);
%         boxes{n} = det{n}.box(valid_score_box, :);
%         vp_labels{n} = vp_labels{n}(valid_score_box);
%     end
    
    if mod(n, 50) == 0
        fprintf(' %d', n);
    end
end % n: image index
fprintf('\n');
assert(n_obj==img_num);


%% evaluate detection performance
fprintf(' evaluate P-R curve and AP ...\n');

Eval.overlap_type = 'iou';
Eval.ov_thresh = 0.5;

% sort detections by decreasing confidence
scores = cell2mat(scores);    % ~ [num_bbox_tot 1]
boxes = cell2mat(boxes)';     % ~ [4 num_bbox_tot]
img_ids = cell2mat(img_ids)'; % ~ [num_bbox_tot 1]
img_ids_all = cell2mat(img_ids_all);

[~, si] = sort(-scores);
img_ids = img_ids(si);
img_ids_all = img_ids_all(si);
boxes = boxes(:, si);

% assign detections to ground truth objects
nd = length(scores);
tp = zeros([nd, 1]);
fp = zeros([nd, 1]);
d1=0;
%%

for d = d1+1: nd
    % display progress
    
    % find ground truth image
    i = img_ids(d);   % i: image id
    
    if i == 0
        fp(d) = 1;
    else
        
        % assign detection to ground truth object if any
        bb = boxes(:, d);
        ovmax = -inf;
        jmax = [];
        for j = 1: size(gt(i).bbox, 2)
            bbgt = gt(i).bbox(:, j);
            bi = [max(bb(1), bbgt(1)); max(bb(2), bbgt(2)); min(bb(3), bbgt(3)); min(bb(4), bbgt(4))];   % intersection
            iw = bi(3) - bi(1) + 1;
            ih = bi(4) - bi(2) + 1;
            if (iw > 0) && (ih > 0)                
                % compute overlap as area of intersection / area of union
                ua = (bb(3) - bb(1) + 1) * (bb(4) - bb(2) + 1) + ...
                     (bbgt(3) - bbgt(1) + 1) * (bbgt(4) - bbgt(2) + 1) - ...
                      iw * ih;    % area of union
                ov = iw * ih / ua;
                if ov > ovmax
                    ovmax = ov;
                    jmax = j;
                end
            end
        end
        
        % assign detection as true positive/don't care/false positive
        if ovmax >= Eval.ov_thresh
            if ~gt(i).diff(jmax)
                if ~gt(i).det(jmax)
                    tp(d) = 1;            % true positive
                    gt(i).det(jmax) = true;
                else
                    fp(d) = 1;            % false positive (multiple/duplicate detection)
                end
            end
        else
            fp(d)=1;                    % false positive
        end
    end % if i == 0
    
    if fp(d) == 1
        d1 = d;
        break
    end
end

n=img_ids(d);
n2 = img_ids_all(d)
bbox = boxes(:,d);

if n>0
    img_name = img_list{1}{n};
    img_name2 = img_list_all{1}{n2};
    assert(strcmp(img_name, img_name2));
    assert(strcmp(img_list_all{3}{n2}, model_category));
    file_img = sprintf('%s/%s.JPEG', dir_img, img_name);
    img = imread(file_img);
    img1 = img(bbox(2):bbox(4), bbox(1):bbox(3),:);
    
    gtbox=gt(n).bbox;
    img2 = img(gtbox(2):gtbox(4), gtbox(1):gtbox(3),:);
    imshowpair(img1, img2, 'montage');
    
    ov=0;
    bi = [max(bbox(1), gtbox(1)); max(bbox(2), gtbox(2)); min(bbox(3), gtbox(3)); min(bbox(4), gtbox(4))];   % intersection
    iw = bi(3) - bi(1) + 1;
    ih = bi(4) - bi(2) + 1;
    if (iw > 0) && (ih > 0)                
        ua = (bbox(3) - bbox(1) + 1) * (bbox(4) - bbox(2) + 1) + ...
                         (gtbox(3) - gtbox(1) + 1) * (gtbox(4) - gtbox(2) + 1) - ...
                          iw * ih;    % area of union
        ov = iw * ih / ua;
    end
    ov
else
    img_name = img_list_all{1}{n2};
    dir_img_oo = sprintf(Dataset.img_dir, img_list_all{3}{n2});
    file_img = sprintf('%s/%s.JPEG', dir_img_oo, img_name);
    img = imread(file_img);
    img1 = img(bbox(2):bbox(4), bbox(1):bbox(3),:);
    imshowpair(img, img1, 'montage');
    
end
end