function crop_pascal_images(category, set_type)
dataset_suffix = 'mergelist_rand';
Dataset.img_dir = '/media/zzs/SSD1TB/zzs/dataset/PASCAL3D+_release1.1/Images/%s_imagenet/';
Dataset.anno_dir = '/media/zzs/SSD1TB/zzs/dataset/PASCAL3D+_release1.1/Annotations/%s_imagenet/';

dir_img = sprintf(Dataset.img_dir, category);
dir_img_save = sprintf('/media/zzs/4TB/qingliu/PASCAL3D+_cropped/%s_imagenet/', category);
if ~exist(dir_img_save, 'dir')
    mkdir(dir_img_save);
end

dir_anno = sprintf(Dataset.anno_dir, category);

Data.gt_dir = './intermediate/ground_truth_data/';
Dataset.train_list = fullfile(Data.gt_dir, ['%s_' sprintf('%s_train.txt', dataset_suffix)]);
Dataset.test_list =  fullfile(Data.gt_dir, ['%s_' sprintf('%s_test.txt', dataset_suffix)]);

switch set_type
    case 'train'
        file_list = sprintf(Dataset.train_list, category);
    case 'test'
        file_list = sprintf(Dataset.test_list, category);
    otherwise
        error('Error: unknown set_type');
end

assert(exist(file_list, 'file') > 0);
file_ids = fopen(file_list, 'r');
img_list = textscan(file_ids, '%s %d');
img_num = length(img_list{1});

caffe_dim = 224;

for n = 1: img_num                       % for each image
    file_img = sprintf('%s/%s.JPEG', dir_img, img_list{1}{n});
    file_img_save = sprintf('%s/%s_%d.JPEG', dir_img_save, img_list{1}{n}, img_list{2}(n));
    img = imread(file_img);
    [height, width, ~] = size(img);
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end

    file_anno = sprintf('%s/%s.mat', dir_anno, img_list{1}{n});
    anno = load(file_anno);
    anno = anno.record;
    bbox = anno.objects(img_list{2}(n)).bbox;
    
    bbox = [max(ceil(bbox(1)), 1), max(ceil(bbox(2)), 1), min(floor(bbox(3)), width), min(floor(bbox(4)), height)];
    patch = img(bbox(2): bbox(4), bbox(1): bbox(3), :);
    scaled_patch = myresize(patch, caffe_dim, 'short');
    
    imwrite(scaled_patch, file_img_save);
    
    if mod(n,100) == 0
        disp(n);
    end
end

end
