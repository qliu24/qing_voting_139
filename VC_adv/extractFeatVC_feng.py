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
from utils import *

dict4 = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_car_VGG16_pool4_K200_vMFMM30.pickle'
dict3 = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_car_VGG16_pool3_K200_vMFMM30.pickle'

with open(dict4, 'rb') as fh:
    _, centers4, _ = pickle.load(fh)
    
with open(dict3, 'rb') as fh:
    _, centers3, _ = pickle.load(fh)


vc4_num = centers4.shape[0]
vc3_num = centers3.shape[0]

thres4 = 0.39
thres3 = 0.31
cache_dir = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'
extractor = FeatureExtractor(cache_folder = cache_dir, layer_names = ['pool3/MaxPool:0', 'pool4/MaxPool:0'])

adv_file_dir = '/export/home/qliu24/VC_adv_data/feng/'


ff_adv = h5py.File(adv_file_dir+'adversial_samples_vc4_vMFMM_2000.mat')
ff_ori = h5py.File(adv_file_dir+'original_samples_vc4_vMFMM_2000.mat')
vc_i = 0
for vc in range(0,200,10):
    adv_images = np.array(ff_adv[ff_adv['adversial_samples'][0][vc_i]].value).astype(float)
    adv_images = np.transpose(adv_images, [0,2,3,1])
    
    features, _ = extractor.extract_from_images(adv_images)
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
    target = open(adv_file_dir+'feng_vc{}_fake'.format(vc), 'w')
    target.write(temp)
    target.close()
    
    ori_images = np.array(ff_ori[ff_ori['original_samples'][0][vc_i]].value).astype(float)
    ori_images = np.transpose(ori_images, [0,2,3,1])
    
    features, _ = extractor.extract_from_images(ori_images)
    pool4 = features[1][:, :, :, :]
    ori4 = normalize_features(pool4.reshape(-1,512))
    
    ori_dis4 = cdist(ori4, centers4, 'cosine')
    ori_code4 = ori_dis4 < thres4
    
    temp2 = json.dumps({'ori4': ori4.tolist(), 'ori_dis4':ori_dis4.tolist(), 'ori_code4': ori_code4.tolist()})
    target2 = open(adv_file_dir+'feng_vc{}_fake_ori'.format(vc), 'w')
    target2.write(temp2)
    target2.close()
    
    vc_i += 1







