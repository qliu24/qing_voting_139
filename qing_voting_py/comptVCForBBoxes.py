import numpy as np
import sys
import glob
import math
import pickle
import scipy.io as sio
from scipy.spatial.distance import cdist
from FeatureExtractor import *
from config_voting import *

def comptVCForBBoxes(category_ls, set_type):
    assert(os.path.isfile(Dictionary))
    with open(Dictionary, 'rb') as fh:
        _,centers = pickle.load(fh)
        
    assert(centers.shape[0]==VC['num'])
    
    for category in category_ls:
        
        dir_img = Dataset['img_dir'].format(category)
        dir_anno = Dataset['anno_dir'].format(category)
        
        file_list = Dataset['{0}_list'.format(set_type)].format(category)
        assert(os.path.isfile(file_list))
        
        with open(file_list, 'r') as fh:
            content = fh.readlines()
        
        img_list = [x.strip().split() for x in content]
        img_num = len(img_list)
        print('total number of images for {1}: {0}'.format(img_num, category))
        
        file_nm='props_feat_{0}_{1}_{2}_*.pickle'.format(category, dataset_suffix, set_type)
        num_batch = len(glob.glob(os.path.join(Feat['cache_dir'], file_nm)))
        
        for ii in range(num_batch):
            file_nm='props_feat_{0}_{1}_{2}_{3}.pickle'.format(category, dataset_suffix, set_type, ii)
            file_cache_feat_batch = os.path.join(Feat['cache_dir'], file_nm)
            with open(file_cache_feat_batch, 'rb') as fh:
                feat = pickle.load(fh)
            
            for cnt_img in range(len(feat)):
                feat_set = feat[cnt_img]['feat']
                feat[cnt_img]['r'] = [None for jj in range(len(feat_set))]
                for jj in range(len(feat_set)):
                    layer_feature = feat_set[jj]
                    height, width = layer_feature.shape[0:2]
                    assert(featDim == layer_feature.shape[2])
                    layer_feature = layer_feature.reshape(-1, featDim)
                    feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
                    layer_feature = layer_feature/feat_norm
                    
                    dist = cdist(layer_feature, centers).reshape(height,width,-1)
                    assert(dist.shape[2]==centers.shape[0]);
                    feat[cnt_img]['r'][jj] = dist
                    
                del(feat[cnt_img]['feat'])
                if cnt_img%10 == 0:
                    print(cnt_img, end=' ')
                    sys.stdout.flush()
            
            print('\n')
            file_nm='props_vc_{0}_{1}_{2}_{3}.pickle'.format(category, dataset_suffix, set_type, ii)
            file_cache_vc_batch = os.path.join(Feat['cache_dir'], file_nm)
            
            with open(file_cache_vc_batch, 'wb') as fh:
                pickle.dump(feat, fh)



