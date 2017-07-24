import numpy as np
import math
import pickle
from scipy.spatial.distance import cdist
import scipy.io as sio
import os
import cv2
from myresize import myresize
from FeatureExtractor_full import *

scale_size = 224
featDim = 512

savefile = '/export/home/qliu24/VC_adv_data/cihang/adv_cls_patches/pool4FeatVC_aeroplane.pickle'

model_cache_folder = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'
extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer='pool4')

'''
Dictionary_car = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_car_VGG16_pool4_K200_vMFMM30.pickle'

with open(Dictionary_car, 'rb') as fh:
    _,centers,_ = pickle.load(fh)
    
'''
dir_anno = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Annotations/aeroplane_imagenet/'
# dir_mat = '/export/home/qliu24/VC_adv_data/cihang/adv_cls_patches/adv_mat_file/'
dir_img = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Images/aeroplane_imagenet/'
file_list = '/export/home/qliu24/VC_adv_data/cihang/adv_cls_patches/aeroplane_mergelist_rand_train.txt'
with open(file_list, 'r') as fh:
    content = fh.readlines()
    
img_list = [x.strip().split() for x in content]
img_num = len(img_list)
print('total number of images: {0}'.format(img_num))

feat_set_ori = [None for nn in range(img_num)]
# r_set_ori = [None for nn in range(img_num)]
# feat_set_fake = [None for nn in range(img_num)]
# r_set_fake = [None for nn in range(img_num)]
for nn in range(img_num):
    # file_mat = os.path.join(dir_mat, '{0}.mat'.format(img_list[nn][0]))
    # assert(os.path.isfile(file_mat))
    # matcontent = sio.loadmat(file_mat)
    # im_ori = matcontent['im_ori'][:,:,[2,1,0]]
    # im_fake = matcontent['im_fool'][:,:,[2,1,0]]
    file_img = os.path.join(dir_img, '{0}.JPEG'.format(img_list[nn][0]))
    assert(os.path.isfile(file_img))
    im_ori = cv2.imread(file_img)
    im_ori = myresize(im_ori, scale_size, 'short')
    
    height, width = im_ori.shape[0:2]
    
    file_anno = os.path.join(dir_anno, '{0}.mat'.format(img_list[nn][0]))
    assert(os.path.isfile(file_anno))
    mat_contents = sio.loadmat(file_anno)
    record = mat_contents['record']
    width_im,height_im = record['imgsize'][0][0][0][0:2]
    objects = record['objects']
    bbox = objects[0,0]['bbox'][0,int(img_list[nn][1])-1][0]
    
    ratio = 224/min(width_im,height_im)
    bbox2 = np.array(bbox)*ratio
    bbox2 = [max(math.ceil(bbox2[0]), 1), max(math.ceil(bbox2[1]), 1), \
             min(math.floor(bbox2[2]), width), min(math.floor(bbox2[3]), height)]
    # print(im_ori.shape)
    patch_ori = im_ori[bbox2[1]-1: bbox2[3], bbox2[0]-1: bbox2[2], :]
    # print(patch_ori.shape)
    try:
        patch_ori = myresize(patch_ori, scale_size, 'short')
    except:
        print(nn, im_ori.shape, bbox, bbox2)
    # print(patch_ori.shape)
    
    # patch_fake = im_fake[bbox2[1]-1: bbox2[3], bbox2[0]-1: bbox2[2], :]
    # patch_fake = myresize(patch_fake, scale_size, 'short')
    
    layer_feature_ori = extractor.extract_feature_image(patch_ori)[0]
    iheight, iwidth = layer_feature_ori.shape[0:2]
    feat_set_ori[nn] = layer_feature_ori
    '''
    layer_feature_ori = layer_feature_ori.reshape(-1, featDim)
    layer_feature_ori = layer_feature_ori/np.sqrt(np.sum(layer_feature_ori**2, 1)).reshape(-1,1)
    dist_ori = cdist(layer_feature_ori, centers, 'cosine').reshape(iheight,iwidth,-1)
    r_set_ori[nn] = dist_ori
    
    
    layer_feature_fake = extractor.extract_feature_image(patch_fake)[0]
    iheight, iwidth = layer_feature_fake.shape[0:2]
    feat_set_fake[nn] = layer_feature_fake
    layer_feature_fake = layer_feature_fake.reshape(-1, featDim)
    layer_feature_fake = layer_feature_fake/np.sqrt(np.sum(layer_feature_fake**2, 1)).reshape(-1,1)
    dist_fake = cdist(layer_feature_fake, centers, 'cosine').reshape(iheight,iwidth,-1)
    r_set_fake[nn] = dist_fake
    '''
    if nn%100 == 0:
        print(nn, end=' ', flush=True)
    
        
print('\n')

with open(savefile, 'wb') as fh:
    # pickle.dump([feat_set_ori, r_set_ori, feat_set_fake, r_set_fake], fh)
    pickle.dump(feat_set_ori, fh)
