function generate_res_info_files_bg(set_type, config_file)
try
    eval(config_file);
catch
    keyboard;
end

switch set_type
    case 'train'
        file_cache_VC_data = fullfile(VC.cache_dir, sprintf('%s_%s_train_bg.mat', category, dataset_suffix));
    case 'test'
        file_cache_VC_data = fullfile(VC.cache_dir, sprintf('%s_%s_test_bg.mat', category, dataset_suffix));
    otherwise
        error('Error: unknown set_type;\n');
end  

assert(exist(file_cache_VC_data, 'file') > 0);
load(file_cache_VC_data, 'r_set');
assert(exist('r_set', 'var')>0);

res_info = cell(1,length(r_set));
for n=1:length(r_set)
    res_info{n}.layer_feature_dist = r_set{n};
    if mod(n,100) == 0
        disp(n);
    end 
end

VC.res_info = '/media/zzs/4TB/qingliu/qing_intermediate/bkmb_K208_res_info/res_info_%s_%s_bg.mat';
save(sprintf(VC.res_info, category, set_type), 'res_info', '-v7.3');

end % end of function
