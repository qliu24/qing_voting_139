load('/media/zzs/4TB/qingliu/qing_intermediate/dictionary_imagenet_bkmb_vgg16_pool4_K208_norm_nowarp_prune_512.mat')
file_dir = '/media/zzs/4TB/qingliu/qing_intermediate/patch_K208_bkmb_pool4/';
MkdirIfMissing(file_dir);
cd(file_dir);
K = 208;
num = 100;
for k = 1:K
    dirnm = sprintf('example_K%d',k);
    mkdir(dirnm);
    for i = 1:num
        imwrite(reshape(example{k}(:,i), 100,100,3), sprintf('example_K%d/%d.jpeg', k, i));
    end
end
