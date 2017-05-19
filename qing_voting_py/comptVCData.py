import math
import pickle
import scipy.io as sio
from scipy.spatial.distance import cdist
from FeatureExtractor import *
from config_voting import *

def comptVCData(category_ls, set_type):
    for category in category_ls:
        file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}.pickle'.format(category, dataset_suffix, set_type))
        assert(os.path.isfile(file_cache_feat))
        with open(file_cache_feat, 'rb') as fh:
            feat_set = pickle.load(fh)
        
        assert(os.path.isfile(Dictionary))
        with open(Dictionary, 'rb') as fh:
            _,centers = pickle.load(fh)
            
        assert(centers.shape[0]==VC['num'])
        print('compute and cache VC distance data for {1} {0} set ...\n'.format(set_type, category))
        
        r_set = [None for nn in range(len(feat_set))]
        for nn in range(len(feat_set)):
            layer_feature = feat_set[nn]
            height = layer_feature.shape[0]
            width = layer_feature.shape[1]
            assert(featDim == layer_feature.shape[2])
            layer_feature = layer_feature.reshape(-1, featDim)
            feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
            layer_feature = layer_feature/feat_norm
            
            dist = cdist(layer_feature, centers).reshape(height,width,-1)
            assert(dist.shape[2]==centers.shape[0]);
            r_set[nn] = dist
            
            if nn%100 == 0:
                print(nn, end=' ')
            
            
        print('\n')
        
        with open(file_cache_feat, 'wb') as fh:
            pickle.dump([feat_set, r_set], fh)