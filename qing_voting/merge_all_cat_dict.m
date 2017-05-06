% merge the features from different objects

function merge_all_cat_dict(config_file, samp_size)

try
    eval(config_file)
catch
    keyboard
end

object = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
save_dir = '/media/zzs/4TB/qingliu/qing_intermediate/dictionary_imagenet_%s_vgg16_%s_nowarp.mat';
save_path = sprintf(save_dir, 'all', layer_name);

img_set_all = cell(1,0);
feat_set_all = single.empty(featDim,0);
loc_set_all = single.empty(5,0);
cnt_img=0;
for i = 1:numel(object)
    category = object{i}; % set the object of interest
    eval(config_file);
    fprintf(' %s', category)
    
    load(sprintf(Dictionary.feature_cache_dir, category, layer_name));
    % roughly balance the training example number across different category
    if length(img_set) >= 1000
        idx = randperm(length(img_set), 1000);
    else
        idx = randi(length(img_set), 1, 1000);
    end
    
    idx2 = arrayfun(@(x) (x-1)*samp_size+1:x*samp_size, idx, 'un',0);
    idx2 = cell2mat(idx2);
    img_set = img_set(idx);
    feat_set = feat_set(:,idx2);
    loc_set = loc_set(:, idx2);
    
    img_set_all = [img_set_all, img_set];
    feat_set_all = cat(2, feat_set_all, feat_set);
    
    idx3 = arrayfun(@(x) ones(1, samp_size)*x, 1:1000, 'un',0);
    idx3 = cell2mat(idx3);
    assert(size(loc_set,2)==length(idx3));
    loc_set(1,:) = idx3+cnt_img;
    % cnt_img = max(loc_set(1,:));
    cnt_img = cnt_img + 1000;
    loc_set_all = cat(2, loc_set_all, loc_set);
    
    assert(cnt_img == length(img_set_all))
end

save(save_path, 'feat_set_all', 'img_set_all', 'loc_set_all', '-v7.3');

end % end of function

