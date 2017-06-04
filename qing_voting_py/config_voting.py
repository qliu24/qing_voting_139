import cv2,os,glob,pickle,sys,math,time
import numpy as np
from myresize import myresize

all_categories = ['car','aeroplane','bicycle','bus','motorbike','train']
all_bgs = ['bg1','bg2','bg3','bg4','bg5','bg6']

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

scale_size = 224
patch_size = [7,7]

Dict = dict()
Dict['file_list'] = '/home/candy/qing_voting_139/qing_voting_py/data/file_list.txt'
Dict['cache_path'] = '/home/candy/qing_voting_139/qing_voting_py/data/pool4_all_dumped_data'

dataset_suffix = 'mergelist_rand'

Dataset = dict()
Dataset['img_dir'] = '/mnt/4T-HD/qing/PASCAL3D+_release1.1/Images/{0}_imagenet/'
Dataset['anno_dir'] = '/mnt/4T-HD/qing/PASCAL3D+_release1.1/Annotations/{0}_imagenet/'

Dataset['gt_dir'] = '/home/candy/qing_voting_139/qing_voting/intermediate/ground_truth_data/'
Dataset['train_list'] = os.path.join(Dataset['gt_dir'], '{0}_'+ '{0}_train.txt'.format(dataset_suffix))
Dataset['test_list'] = os.path.join(Dataset['gt_dir'], '{0}_'+ '{0}_test.txt'.format(dataset_suffix))

model_cache_folder = '/home/candy/Siamese_iclr17_tf/cache/'

Data=dict()
Data['root_dir'] = '/home/candy/qing_voting_139/qing_voting/intermediate/data'
Data['root_dir2'] = '/mnt/4T-HD/qing/intermediate/'

Feat = dict()
Feat['cache_dir'] = os.path.join(Data['root_dir2'], 'feat')
if not os.path.exists(Feat['cache_dir']):
    os.makedirs(Feat['cache_dir'])
    
Feat['cache_dir2'] = os.path.join(Data['root_dir2'], 'feat2')
if not os.path.exists(Feat['cache_dir2']):
    os.makedirs(Feat['cache_dir2'])
    
Feat['num_batch_img'] = 100
Feat['max_num_props_per_img'] = 150


VC = dict()
VC['num'] = 190
VC['num_car'] = 158
VC['num_super'] = 80
VC['layer'] = 'pool4'
if not os.path.exists(os.path.join(Data['root_dir2'], 'feat_pickle')):
    os.makedirs(os.path.join(Data['root_dir2'], 'feat_pickle'))
    
VC['res_info'] = os.path.join(Data['root_dir2'], 'feat_pickle', 'res_info_{0}_{1}.pickle')

Dictionary = '/home/candy/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_VGG16_{0}_K{1}_prune_{2}.pickle'.format(VC['layer'], VC['num'], featDim)

Dictionary_car = '/home/candy/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_car_VGG16_{0}_K{1}_prune_{2}.pickle'.format(VC['layer'], VC['num_car'], featDim)

Dictionary_super = os.path.join(Feat['cache_dir'], 'dictionary_super_PASCAL3D+_VGG16_{0}_K{1}.pickle'.format(VC['layer'], VC['num_super']))

Dictionary_super_car = os.path.join(Feat['cache_dir'], 'dictionary_super_PASCAL3D+_car_VGG16_{0}_K{1}.pickle'.format(VC['layer'], VC['num_super']))

nms_thrh = 0.3
Eval_ov_thrh = 0.5
Eval = dict()
Eval['dist_thresh'] = 56
Eval['iou_thresh'] = 0.5

dir_det_result = os.path.join(Data['root_dir2'], 'result')
if not os.path.exists(dir_det_result):
    os.makedirs(dir_det_result)
    
dir_perf_eval = os.path.join(Data['root_dir2'], 'eval_obj_det')
if not os.path.exists(dir_perf_eval):
    os.makedirs(dir_perf_eval)
    
# file_det_result = os.path.join(dir_det_result, 'props_det_{0}_{1}_{2}_{3}_{4}'.format(model_category, category, dataset_suffix, set_type, model_suffix))

Model_dir = os.path.join(Data['root_dir2'], 'unary_weights')

model_file_dic = dict()
for category in ['car','aeroplane','bicycle','bus','motorbike','train']:
    model_file_dic[category] = os.path.join(Model_dir, '{0}_K4_softstart.pickle'.format(category))
    
model_file_dic['bg1'] = os.path.join(Model_dir, 'car_train_bg.pickle')
model_file_dic['bg2'] = os.path.join(Model_dir, 'aeroplane_train_bg.pickle')
model_file_dic['bg3'] = os.path.join(Model_dir, 'bicycle_train_bg.pickle')
model_file_dic['bg4'] = os.path.join(Model_dir, 'bus_train_bg.pickle')
model_file_dic['bg5'] = os.path.join(Model_dir, 'motorbike_train_bg.pickle')
model_file_dic['bg6'] = os.path.join(Model_dir, 'train_train_bg.pickle')

model_dim_dic = dict()
model_dim_dic['car'] = [80, 24, 5]
model_dim_dic['aeroplane'] = [80, 25, 8]
model_dim_dic['bicycle'] = [80, 13, 12]
model_dim_dic['bus'] = [80, 23, 9]
model_dim_dic['motorbike'] = [80, 14, 15]
model_dim_dic['train'] = [80, 25, 14]
# model_dim_dic['car'] = [80, 48, 10]
# model_dim_dic['aeroplane'] = [80, 49, 15]
# model_dim_dic['bicycle'] = [80, 26, 23]
# model_dim_dic['bus'] = [80, 45, 18]
# model_dim_dic['motorbike'] = [80, 28, 30]
# model_dim_dic['train'] = [80, 49, 28]

SP = dict()
SP['img_list'] = '/mnt/4T-HD/qing/intermediate_sp/dataset/test_list/{0}_test.txt'
SP['anno_dir'] = '/mnt/4T-HD/qing/intermediate_sp/SP_final/{0}_imagenet/transfered'
SP['feat_dir'] = '/mnt/4T-HD/qing/intermediate_sp/feat/'
SP['result'] = '/mnt/4T-HD/qing/intermediate_sp/result/'
