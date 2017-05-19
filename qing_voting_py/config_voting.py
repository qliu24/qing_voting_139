import os
import numpy as np

Apad_set = [2, 6, 18, 42, 90] # padding size
Astride_set = [2, 4, 8, 16, 32] # stride size
featDim_set = [64, 128, 256, 512, 512] # feature dimension
Arf_set = [6, 16, 44, 100, 212]
offset_set = np.ceil(np.array(Apad_set).astype(float)/np.array(Astride_set)).astype(int)

# get pool4 layer parameters
pool4_n = 3
Apad = Apad_set[pool4_n]
Astride = Astride_set[pool4_n]
featDim = featDim_set[pool4_n]
Arf = Arf_set[pool4_n]
offset = offset_set[pool4_n]


dataset_suffix = 'mergelist_rand'

Dataset = dict()
Dataset['img_dir'] = '/mnt/4T-HD/qing/PASCAL3D+_release1.1/Images/{0}_imagenet/'
Dataset['anno_dir'] = '/mnt/4T-HD/qing/PASCAL3D+_release1.1/Annotations/{0}_imagenet/'

Dataset['gt_dir'] = '../qing_voting/intermediate/ground_truth_data/'
Dataset['train_list'] = os.path.join(Dataset['gt_dir'], '{0}_'+ '{0}_train.txt'.format(dataset_suffix))
Dataset['test_list'] = os.path.join(Dataset['gt_dir'], '{0}_'+ '{0}_test.txt'.format(dataset_suffix))

model_cache_folder = '/home/candy/Siamese_iclr17_tf/cache/'

Feat = dict()
Feat['cache_dir'] = '/mnt/4T-HD/qing/intermediate/feat/'

VC = dict()
VC['num'] = 198
VC['layer'] = 'pool4'
VC['res_info'] = '/mnt/4T-HD/qing/intermediate/feat_pickle/res_info_{0}_{1}.pickle'

Dictionary = '/mnt/4T-HD/qing/voting_data/VC_siamese/dictionary_PASCAL3D+_VGG16_{0}_K{1}_prune_{2}.pickle'.format(VC['layer'], VC['num'], featDim)
