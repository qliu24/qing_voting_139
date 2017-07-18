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

def normalize_features(features):
    '''features: n by d matrix'''
    assert(len(features.shape)==2)
    return features/np.sqrt(np.sum(features**2, axis=1).reshape(-1,1))

dict4 = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_imagenet_car_vgg16_pool4_K176_norm_nowarp_prune_512.pickle'
dict3 = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_imagenet_car_vgg16_pool3_K223_norm_nowarp_prune_512.pickle'

with open(dict4, 'rb') as fh:
    _, centers4 = pickle.load(fh)
    
with open(dict3, 'rb') as fh:
    _, centers3 = pickle.load(fh)
    
centers4 = normalize_features(centers4)
centers3 = normalize_features(centers3)

vc4_num = centers4.shape[0]
vc3_num = centers3.shape[0]
cihang_mean_RGB = np.float32([[[122.7717, 115.9465, 102.9801]]])

thres4 = 0.42
thres3 = 0.31
cache_dir = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'
extractor = FeatureExtractor(cache_folder = cache_dir, layer_names = ['pool3/MaxPool:0', 'pool4/MaxPool:0'])

adv_file_dir = '/export/home/qliu24/VC_adv_data/cihang/'

image_list = []
image_list_ori = []
vc_ori_list = []
for n in range(1, 2000):
    file = sio.loadmat(adv_file_dir+'adv_patches/{}.mat'.format(n*5))
    image = np.array(file['im_fool'], dtype=np.float32)
    image_ori = np.array(file['im_ori'], dtype=np.float32)
    vc_ori = np.array(file['vc_idx_ori'], dtype=np.int32)
    vc_ori_list.append(deepcopy(vc_ori))
    image = np.expand_dims(image, axis=0)
    image_ori = np.expand_dims(image_ori, axis=0)
    image_list.append(deepcopy(image))
    image_list_ori.append(deepcopy(image_ori))

vc_ori = np.concatenate(vc_ori_list)
images = np.concatenate(image_list, axis=0)
images -= cihang_mean_RGB
images_ori = np.concatenate(image_list_ori, axis=0)
images_ori -= cihang_mean_RGB

features, _ = extractor.extract_from_images(images)
pool4 = features[1][:, :, :, :]
fake4 = normalize_features(pool4.reshape(-1,512))

pool3 = features[0][:, :, :, :]
fake3 = normalize_features(pool3.reshape(-1,256))

fake_dis4 = cdist(fake4, centers4, 'cosine')
fake_dis3 = cdist(fake3, centers3, 'cosine').reshape(-1,8,8,vc3_num)
assert(fake_dis4.shape[0]==fake_dis3.shape[0])

fake_code4 = fake_dis4 < thres4
fake_code3 = fake_dis3 < thres3

temp = json.dumps({'fake_code4': fake_code4.tolist(), 'fake_code3': fake_code3.tolist()})
target = open(adv_file_dir+'cihang_vc1_fake', 'w')
target.write(temp)
target.close()

features, _ = extractor.extract_from_images(images_ori)
pool4 = features[1][:, :, :, :]
ori4 = normalize_features(pool4.reshape(-1,512))

ori_dis4 = cdist(ori4, centers4, 'cosine')
ori_code4 = ori_dis4 < thres4

temp2 = json.dumps({'ori4': ori4.tolist(), 'ori_dis4':ori_dis4.tolist(), 'ori_code4': ori_code4.tolist()})
target2 = open(adv_file_dir+'cihang_vc1_fake_ori', 'w')
target2.write(temp2)
target2.close()
