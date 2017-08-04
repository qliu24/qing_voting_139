import numpy as np
import math
import pickle
from scipy.spatial.distance import cdist
import scipy.io as sio
import os
import cv2
from myresize import myresize
from FeatureExtractor_full import *


dataset_suffix = 'mergelist_rand'
Dataset = dict()
Dataset['img_dir'] = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Images/{0}_imagenet/'
Dataset['anno_dir'] = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Annotations/{0}_imagenet/'
Dataset['gt_dir'] = '/export/home/qliu24/qing_voting_139/qing_voting_py/intermediate/ground_truth_data/'
Dataset['train_list'] = os.path.join(Dataset['gt_dir'], '{0}_'+ '{0}_train.txt'.format(dataset_suffix))
Dataset['test_list'] = os.path.join(Dataset['gt_dir'], '{0}_'+ '{0}_test.txt'.format(dataset_suffix))
Dataset['adv_dir'] = '/export/home/qliu24/VC_adv_data/qing/VGG_adv/{}_{}'

cache_folder = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'

######### config #############
category = 'car'
target = 'bus'
set_type= 'test'
# dir_img = Dataset['img_dir'].format(category)
dir_anno = Dataset['anno_dir'].format(category)
dir_adv = Dataset['adv_dir'].format(category, target)

file_list = Dataset['{}_list'.format(set_type)].format(category)


scale_size = 224
featDim = 512

savefile = '/export/home/qliu24/VC_adv_data/qing/VGG_adv/feat/pool4FeatVC_adv_{}_{}.pickle'.format(category, target)

############### init extractor
extractor = FeatureExtractor(cache_folder=cache_folder, which_net='vgg16', which_layer='pool4')


############### read-in images
with open(file_list, 'r') as fh:
    content = fh.readlines()
    
img_list = [x.strip().split() for x in content]
img_num = len(img_list)
print('total number of images: {0}'.format(img_num))

feat_set_ori = [None for nn in range(img_num)]
# r_set_ori = [None for nn in range(img_num)]
feat_set_fool = [None for nn in range(img_num)]
# r_set_fool = [None for nn in range(img_num)]
for nn in range(img_num):
    file_adv = os.path.join(dir_adv, '{0}.pickle'.format(img_list[nn][0]))
    assert(os.path.isfile(file_adv))
    with open(file_adv, 'rb') as fh:
        im_ori, _, im_fool = pickle.load(fh)
    
    '''
    file_img = os.path.join(dir_img, '{0}.JPEG'.format(img_list[nn][0]))
    assert(os.path.isfile(file_img))
    im_ori = cv2.imread(file_img)
    height_ori, width_ori = im_ori.shape[0:2]
    im_ori = myresize(im_ori, scale_size, 'short')
    '''
    height, width = im_ori.shape[0:2]
    
    file_anno = os.path.join(dir_anno, '{0}.mat'.format(img_list[nn][0]))
    assert(os.path.isfile(file_anno))
    mat_contents = sio.loadmat(file_anno)
    record = mat_contents['record']
    width_im,height_im = record['imgsize'][0][0][0][0:2]
    # assert(height_ori == height_im)
    # assert(width_ori == width_im)
    objects = record['objects']
    bbox = objects[0,0]['bbox'][0,int(img_list[nn][1])-1][0]
    
    ratio = 224/min(width_im,height_im)
    bbox2 = np.array(bbox)*ratio
    bbox2 = [max(math.ceil(bbox2[0]), 1), max(math.ceil(bbox2[1]), 1), \
             min(math.floor(bbox2[2]), width), min(math.floor(bbox2[3]), height)]
    
    patch_ori = im_ori[bbox2[1]-1: bbox2[3], bbox2[0]-1: bbox2[2], :]
    
    try:
        patch_ori = myresize(patch_ori, scale_size, 'short')
    except:
        print(nn, im_ori.shape, bbox, bbox2)
    
    
    patch_fool = im_fool[bbox2[1]-1: bbox2[3], bbox2[0]-1: bbox2[2], :]
    patch_fool = myresize(patch_fool, scale_size, 'short')
    
    layer_feature_ori = extractor.extract_feature_image(patch_ori)[0]
    feat_set_ori[nn] = layer_feature_ori
    
    layer_feature_fool = extractor.extract_feature_image(patch_fool)[0]
    feat_set_fool[nn] = layer_feature_fool
    '''
    iheight, iwidth = layer_feature_ori.shape[0:2]
    layer_feature_ori = layer_feature_ori.reshape(-1, featDim)
    layer_feature_ori = layer_feature_ori/np.sqrt(np.sum(layer_feature_ori**2, 1)).reshape(-1,1)
    dist_ori = cdist(layer_feature_ori, centers, 'cosine').reshape(iheight,iwidth,-1)
    r_set_ori[nn] = dist_ori
    
    
    layer_feature_fool = extractor.extract_feature_image(patch_fool)[0]
    iheight, iwidth = layer_feature_fool.shape[0:2]
    feat_set_fool[nn] = layer_feature_fool
    layer_feature_fool = layer_feature_fool.reshape(-1, featDim)
    layer_feature_fool = layer_feature_fool/np.sqrt(np.sum(layer_feature_fool**2, 1)).reshape(-1,1)
    dist_fool = cdist(layer_feature_fool, centers, 'cosine').reshape(iheight,iwidth,-1)
    r_set_fool[nn] = dist_fool
    '''
    if nn%100 == 0:
        print(nn, end=' ', flush=True)
    
        
print('\n')

with open(savefile, 'wb') as fh:
    # pickle.dump([feat_set_ori, r_set_ori, feat_set_fool, r_set_fool], fh)
    pickle.dump([feat_set_ori, feat_set_fool], fh)


