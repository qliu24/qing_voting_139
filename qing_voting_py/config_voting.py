import cv2,os,glob,pickle,sys,math,time
import numpy as np
from myresize import myresize

# all_categories = ['car','aeroplane','bicycle','bus','motorbike','train']
# all_bgs = ['bg1','bg2','bg3','bg4','bg5','bg6']
all_categories = ['car']
all_bgs = ['bg1']

dataset_suffix = 'mergelist_rand'

net_type='VGG'
# Alexnet
if net_type=='alex':
    Apad_set = [0, 0, 16, 16, 32, 48, 64] # padding size
    Astride_set = [4, 8, 8, 16, 16, 16, 16] # stride size
    featDim_set = [96, 96, 256, 256, 384, 384, 256] # feature dimension
    Arf_set = [11, 19, 51, 67, 99, 131, 163]
    offset_set = np.ceil(np.array(Apad_set)/np.array(Astride_set)).astype(int)
    layer_n = 4 # conv3
elif net_type=='VGG':
    Apad_set = [2, 6, 18, 42, 90] # padding size
    Astride_set = [2, 4, 8, 16, 32] # stride size
    featDim_set = [64, 128, 256, 512, 512] # feature dimension
    Arf_set = [6, 16, 44, 100, 212]
    offset_set = np.ceil(np.array(Apad_set).astype(float)/np.array(Astride_set)).astype(int)
    # layer_n = 3 # pool4
    layer_n = 2 # pool3
    # layer_n = 1 # pool2
    
Apad = Apad_set[layer_n]
Astride = Astride_set[layer_n]
featDim = featDim_set[layer_n]
Arf = Arf_set[layer_n]
offset = offset_set[layer_n]

scale_size = 224
patch_size = [7,7]

VC = dict()
VC['num'] = 200
VC['num_car'] = 200
VC['num_super'] = 80

if net_type=='alex':
    VC['layer'] = 'conv3'
elif net_type=='VGG':
    # VC['layer'] = 'pool4'
    VC['layer'] = 'pool3'
    # VC['layer'] = 'pool2'

Dict = dict()
Dict['file_list'] = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/file_list.txt'
Dict['cache_path'] = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/{0}_all_dumped_data'.format(VC['layer'])

Dataset = dict()
Dataset['img_dir'] = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Images/{0}_imagenet/'
Dataset['anno_dir'] = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Annotations/{0}_imagenet/'

Dataset['gt_dir'] = '/export/home/qliu24/qing_voting_139/qing_voting_py/intermediate/ground_truth_data/'
Dataset['train_list'] = os.path.join(Dataset['gt_dir'], '{0}_'+ '{0}_train.txt'.format(dataset_suffix))
Dataset['test_list'] = os.path.join(Dataset['gt_dir'], '{0}_'+ '{0}_test.txt'.format(dataset_suffix))

model_cache_folder = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'

Data=dict()
Data['root_dir'] = '/export/home/qliu24/qing_voting_139/qing_voting_py/intermediate/data'
Data['root_dir2'] = '/export/home/qliu24/qing_voting_data/intermediate/'

Feat = dict()
Feat['cache_dir'] = os.path.join(Data['root_dir2'], 'feat_car')
if not os.path.exists(Feat['cache_dir']):
    os.makedirs(Feat['cache_dir'])
    
# Feat['cache_dir2'] = os.path.join(Data['root_dir2'], 'feat2')
# if not os.path.exists(Feat['cache_dir2']):
#     os.makedirs(Feat['cache_dir2'])
    
Feat['num_batch_img'] = 100
Feat['max_num_props_per_img'] = 150
    
if not os.path.exists(os.path.join(Data['root_dir2'], 'feat_pickle_alex')):
    os.makedirs(os.path.join(Data['root_dir2'], 'feat_pickle_alex'))
    
VC['res_info'] = os.path.join(Data['root_dir2'], 'feat_car', 'res_info_{0}_{1}.pickle')

# Dictionary = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_ALEX_{0}_K{1}_prune_{2}.pickle'.format(VC['layer'], VC['num'], 512)

# Dictionary = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_ALEX_{0}_K{1}_gray.pickle'.format(VC['layer'], VC['num'])
# Dictionary = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_imagenet_car_vgg16_pool4_K176_norm_nowarp_prune_512.pickle'

Dictionary_car = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_car_VGG16_{0}_K{1}_vMFMM30.pickle'.format(VC['layer'], VC['num_car'])

# Dictionary_super = os.path.join(Feat['cache_dir'], 'dictionary_super_PASCAL3D+_VGG16_{0}_K{1}.pickle'.format(VC['layer'], VC['num_super']))

# Dictionary_super_car = os.path.join(Feat['cache_dir'], 'dictionary_super_PASCAL3D+_car_VGG16_{0}_K{1}.pickle'.format(VC['layer'], VC['num_super']))

nms_thrh = 0.3
Eval_ov_thrh = 0.5
Eval = dict()
Eval['dist_thresh'] = 56
Eval['iou_thresh'] = 0.5

dir_det_result = os.path.join(Data['root_dir2'], 'result_car')
if not os.path.exists(dir_det_result):
    os.makedirs(dir_det_result)
    
dir_perf_eval = os.path.join(Data['root_dir2'], 'eval_obj_det')
if not os.path.exists(dir_perf_eval):
    os.makedirs(dir_perf_eval)
    
# file_det_result = os.path.join(dir_det_result, 'props_det_{0}_{1}_{2}_{3}_{4}'.format(model_category, category, dataset_suffix, set_type, model_suffix))

Model_dir = os.path.join(Data['root_dir2'], 'VCpart_model_car')

model_file_dic = dict()
for category in ['car']:
    model_file_dic[category] = os.path.join(Model_dir, 'VCpart_model_{0}_K4_term3_gmthrh9.pickle'.format(category))
    
model_file_dic['bg1'] = os.path.join(Model_dir, 'VCpart_model_car_bg.pickle')
'''
# Model_dir = os.path.join(Data['root_dir2'], 'unary_weights')
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
# model_dim_dic['car'] = [80, 24, 5]
# model_dim_dic['aeroplane'] = [80, 25, 8]
# model_dim_dic['bicycle'] = [80, 13, 12]
# model_dim_dic['bus'] = [80, 23, 9]
# model_dim_dic['motorbike'] = [80, 14, 15]
# model_dim_dic['train'] = [80, 25, 14]

model_dim_dic['car'] = [80, 48, 10]
model_dim_dic['aeroplane'] = [80, 49, 15]
model_dim_dic['bicycle'] = [80, 26, 23]
model_dim_dic['bus'] = [80, 45, 18]
model_dim_dic['motorbike'] = [80, 28, 30]
model_dim_dic['train'] = [80, 49, 28]


model_dim_dic['car'] = [190, 54, 16]
model_dim_dic['aeroplane'] = [190, 55, 21]
model_dim_dic['bicycle'] = [190, 32, 29]
model_dim_dic['bus'] = [190, 51, 24]
model_dim_dic['motorbike'] = [190, 34, 36]
model_dim_dic['train'] = [190, 55, 34]
'''
SP = dict()
SP['img_list'] = '/export/home/qliu24/SP_data/dataset/test_list/{0}_test.txt'
SP['anno_dir'] = '/export/home/qliu24/SP_data/SP_final/{0}_imagenet/transfered'
SP['feat_dir'] = '/export/home/qliu24/SP_data/feat/'
SP['result'] = '/export/home/qliu24/SP_data/result/'
