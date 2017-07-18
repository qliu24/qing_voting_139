import numpy as np
import math
import pickle
from scipy.spatial.distance import cdist
import scipy.io as sio
from FeatureExtractor import *
from config_voting import *

def extractLayerVC(category_ls, set_type):
    assert(os.path.isfile(Dictionary_car))
    with open(Dictionary_car, 'rb') as fh:
        _,centers,_ = pickle.load(fh)
    
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
        
        
        featfile = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}_carVC.pickle'.format(category, dataset_suffix, set_type))
        with open(featfile, 'rb') as fh:
            feat_set, _ = pickle.load(fh)
        
        assert(len(feat_set)==img_num)
        
        r_set = [None for nn in range(img_num)]
        for nn in range(img_num):
            layer_feature = np.copy(feat_set[nn])
            iheight,iwidth = layer_feature.shape[0:2]
            
            layer_feature = layer_feature.reshape(-1, featDim)
            feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
            layer_feature = layer_feature/feat_norm
            
            dist = cdist(layer_feature, centers, 'cosine').reshape(iheight,iwidth,-1)
            assert(dist.shape[2]==centers.shape[0]);
            r_set[nn] = dist
            
            if nn%100 == 0:
                print(nn, end=' ')
                sys.stdout.flush()
            
        print('\n')
        
        file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}_carVC_vMFMM30.pickle'.format(category, dataset_suffix, set_type))
        with open(file_cache_feat, 'wb') as fh:
            pickle.dump([feat_set, r_set], fh)
            
            
if __name__=='__main__':
    # objs = ['car','aeroplane','bicycle','bus','motorbike','train']
    objs = ['car']
    extractLayerVC(objs, 'train')