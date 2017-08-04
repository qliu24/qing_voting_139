import cv2,os,glob,pickle,sys,math,time
import numpy as np
from myresize import myresize

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
    layer_n = 3 # pool4
    # layer_n = 2 # pool3
    # layer_n = 1 # pool2
    
Apad = Apad_set[layer_n]
Astride = Astride_set[layer_n]
featDim = featDim_set[layer_n]
Arf = Arf_set[layer_n]
offset = offset_set[layer_n]

scale_size = 224

VC = dict()
VC['num'] = 200

if net_type=='alex':
    VC['layer'] = 'conv3'
elif net_type=='VGG':
    VC['layer'] = 'pool4'
    # VC['layer'] = 'pool3'
    # VC['layer'] = 'pool2'

model_cache_folder = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'
Dict = dict()
Dict['file_list'] = '/export/home/qliu24/dataset/ILSVRC12/list_fg/file_list.txt'
Dict['cache_path'] = '/export/home/qliu24/ILSVRC12_VC/feat/{0}_all_dumped_data'.format(VC['layer'])
Dict['file_dir'] = '/export/home/qliu24/dataset/ILSVRC12/ILSVRC2012/train_fg/'