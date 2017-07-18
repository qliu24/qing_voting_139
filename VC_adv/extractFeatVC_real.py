import numpy as np
import scipy.io as sio
import os
import cv2
from FeatureExtractor import FeatureExtractor
import h5py
import pickle
import json
from scipy.spatial.distance import cdist
from copy import *
from myresize import myresize
import math
from utils import *

# target vc
vc_ls = list(range(0,200,10))
thres4 = 0.39
thres3 = 0.31
thres3_2 = 0.25
thres2 = 0.21
# Astride = 16
# Apad = 42
# Arf = 100

Astride = 8
Apad = 18
Arf = 44

scale_size=224
feng_mean_RGB = np.float32([[[103.939, 116.779, 123.68]]])

file_path = '/export/home/qliu24/qing_voting_data/intermediate/feat_car/'
filename = file_path + 'car_mergelist_rand_train_car_pool3_vMFMM.pickle'

with open(filename, 'rb') as fh:
    layer_feature, _ = pickle.load(fh)
    
'''
dict4 = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_imagenet_car_vgg16_pool4_K176_norm_nowarp_prune_512.pickle'
dict3 = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_imagenet_car_vgg16_pool3_K223_norm_nowarp_prune_512.pickle'

with open(dict4, 'rb') as fh:
    _, centers4 = pickle.load(fh)
    
with open(dict3, 'rb') as fh:
    _, centers3 = pickle.load(fh)
    
centers4 = normalize_features(centers4)
centers3 = normalize_features(centers3)
'''
dict4 = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_car_VGG16_pool4_K200_vMFMM30.pickle'
dict3 = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_car_VGG16_pool3_K200_vMFMM30.pickle'
dict2 = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_car_VGG16_pool2_K100_vMFMM30.pickle'

with open(dict4, 'rb') as fh:
    _, centers4, _ = pickle.load(fh)
    
with open(dict3, 'rb') as fh:
    _, centers3, _ = pickle.load(fh)
    
with open(dict2, 'rb') as fh:
    _, centers2, _ = pickle.load(fh)
    
vc4_num = centers4.shape[0]
vc3_num = centers3.shape[0]
vc2_num = centers2.shape[0]

# get patches that fired at target vc from natural images
dir_img = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Images/car_imagenet/'
dir_anno = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Annotations/car_imagenet/'
file_list = '/export/home/qliu24/qing_voting_139/qing_voting_py/intermediate/ground_truth_data/car_mergelist_rand_train.txt'

with open(file_list, 'r') as fh:
    content = fh.readlines()

img_list = [x.strip().split() for x in content]
img_num = len(img_list)
assert(img_num == len(layer_feature))
print('total number of images for {1}: {0}'.format(img_num, 'car'))

cache_dir = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'
extractor = FeatureExtractor(cache_folder = cache_dir, layer_names = ['pool2/MaxPool:0', 'pool3/MaxPool:0'])
adv_file_dir = '/export/home/qliu24/VC_adv_data/feng/vc3/'

vc_patch_ls = [[] for _ in vc_ls]
for nn in range(img_num):
    file_img = os.path.join(dir_img, '{0}.JPEG'.format(img_list[nn][0]))
    assert(os.path.isfile(file_img))
    img = cv2.imread(file_img)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    height, width = img.shape[0:2]

    file_anno = os.path.join(dir_anno, '{0}.mat'.format(img_list[nn][0]))
    assert(os.path.isfile(file_anno))
    mat_contents = sio.loadmat(file_anno)
    record = mat_contents['record']
    objects = record['objects']
    bbox = objects[0,0]['bbox'][0,int(img_list[nn][1])-1][0]
    bbox = [max(math.ceil(bbox[0]), 1), max(math.ceil(bbox[1]), 1), \
            min(math.floor(bbox[2]), width), min(math.floor(bbox[3]), height)]
    patch = img[bbox[1]-1: bbox[3], bbox[0]-1: bbox[2], :]
    # patch = cv2.resize(patch, (scale_size, scale_size))
    patch = myresize(patch, scale_size, 'short')
    pheight, pwidth = patch.shape[0:2]
    
    
    iheight,iwidth = layer_feature[nn].shape[0:2]
    lff = layer_feature[nn].reshape(-1, 256)
    lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
    lfd = cdist(lff_norm, centers3, 'cosine').reshape(iheight,iwidth,-1)
    
    for vci, vcii in enumerate(vc_ls):
        rff, cff = np.where(lfd[:,:,vcii] < thres3_2)
        irff = Astride * rff - Apad
        icff = Astride * cff - Apad

        for rii, cii in zip(irff, icff):
            if rii >=0 and rii <= pheight-Arf and cii >=0 and cii <= pwidth-Arf:
                fpatch = np.expand_dims(patch[rii:rii+Arf, cii:cii+Arf, :], axis=0)
                vc_patch_ls[vci].append(fpatch)
                

for vci, vcii in enumerate(vc_ls):
    vpls = np.concatenate(vc_patch_ls[vci], axis=0)
    print(vpls.shape)
    to_save = 2000
    if vpls.shape[0]>to_save:
        vpls = vpls[np.random.permutation(vpls.shape[0])[0:to_save],:,:,:]
    
    images = vpls - feng_mean_RGB
    features, _ = extractor.extract_from_images(images)
    pool4 = features[1][:, :, :, :]
    real4 = normalize_features(pool4.reshape(-1,256))
    
    pool3 = features[0][:, :, :, :]
    real3 = normalize_features(pool3.reshape(-1,128))
    
    real_dis4 = cdist(real4, centers3, 'cosine')
    real_dis3 = cdist(real3, centers2, 'cosine').reshape(-1,8,8,vc2_num)
    assert(real_dis4.shape[0]==real_dis3.shape[0])
    
    real_code4 = real_dis4 < thres3_2
    real_code3 = real_dis3 < thres2
    
    temp = json.dumps({'real_code3': real_code4.tolist(), 'real_code2': real_code3.tolist()})
    target = open(adv_file_dir+'vc{}_real_vc3'.format(vcii), 'w')
    target.write(temp)
    target.close()

