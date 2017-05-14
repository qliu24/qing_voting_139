n=2221;
img = imread(det_all{n}.img_path);
height = det_all{n}.img_siz(1);
width = det_all{n}.img_siz(2);
objects = {'car','bus','aeroplane','bicycle','motorbike','train'};

score1 = score_rst{n};
score2 = score_rst2{n};
bbox = det_all{n}.box;

score1 = score1(nms_list_all2{n}, :);
score2 = score2(nms_list_all2{n}, :);
bbox = bbox(nms_list_all2{n}, :);

score_now = score2;
num_bb=5;

[score1_s, idx_s] = sort(-max(score_now,[],2));
colorls = {'yellow','magenta','cyan','red','green','blue','black'};
for ii = 1:min(num_bb,numel(idx_s))
    idxi = idx_s(ii);
    bbox_i = bbox(idxi, :);
    bbox_i = [max(ceil(bbox_i(1)), 1), max(ceil(bbox_i(2)), 1), min(floor(bbox_i(3)), width), min(floor(bbox_i(4)), height)];
    % rectangle('Position', [bbox(1),bbox(2),bbox(3)-bbox(1),bbox(4)-bbox(2)], 'EdgeColor','r','LineWidth',2);
    [~,si] = max(score_now(idxi,:));
    img = insertText(img, [bbox_i(1)+5,bbox_i(2)+5], sprintf('%s_%f',objects{si},-score1_s(ii)),'FontSize',12);
end

imshow(img);

for ii = 1:min(num_bb,numel(idx_s))
    idxi = idx_s(ii);
    bbox_i = bbox(idxi, :);
    bbox_i = [max(ceil(bbox_i(1)), 1), max(ceil(bbox_i(2)), 1), min(floor(bbox_i(3)), width), min(floor(bbox_i(4)), height)];
    rectangle('Position', [bbox_i(1),bbox_i(2),bbox_i(3)-bbox_i(1),bbox_i(4)-bbox_i(2)], 'EdgeColor',colorls{ii},'LineWidth',2);
    % rectangle('Position', [bbox_i(1),bbox_i(2),bbox_i(3)-bbox_i(1),bbox_i(4)-bbox_i(2)], 'EdgeColor','r','LineWidth',2);
    % [~,si] = max(score1(idxi,:));
    % img = insertText(img, [bbox(1)+5,bbox(2)+5], objects{si},'FontSize',16);
end