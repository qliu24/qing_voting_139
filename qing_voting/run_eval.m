global category model_category Eval;
objects = {'car','bus','aeroplane','bicycle','motorbike','train'};
nms_ls = [0.3 0.2 0.1];
for oi = 1:numel(objects)
    % for ni = 1:numel(nms_ls)
        category='all';
        model_category=objects{oi}
        % Eval.nms_bbox_ratio=nms_ls(ni);
        Eval.nms_bbox_ratio = 0.3;
        evalVotingModelForObjectDetection_all2('config_qing3');
    % end
end
