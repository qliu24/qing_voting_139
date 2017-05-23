import math
import pickle
import scipy.io as sio
from scipy.spatial.distance import cdist
from FeatureExtractor import *
from config_voting import *

def extractLayerFeatSuperVC(category_ls, set_type):
    assert(os.path.isfile(Dictionary_super))
    with open(Dictionary_super, 'rb') as fh:
        _,centers = pickle.load(fh)
        
    assert(centers.shape[0]==VC['num_super'])
    patch_size = [7,7]
    
    for category in category_ls:
        file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}.pickle'.format(category, dataset_suffix, set_type))
        assert(os.path.isfile(file_cache_feat))
        with open(file_cache_feat, 'rb') as fh:
            feat_set, r_set = pickle.load(fh)
            
        print('compute and cache super VC distance for {1} {0} set ...\n'.format(set_type, category))
        
        r_super_set = [None for nn in range(len(r_set))]
        for nn in range(len(r_set)):
            h,w,fdim = r_set[nn].shape
            assert(fdim == VC['num'])
            super_f = np.zeros((h-patch_size[0]+1, w-patch_size[1]+1, patch_size[0]*patch_size[1]*fdim))
            for hh in range(0, h-patch_size[0]+1):
                for ww in range(0, w-patch_size[1]+1):
                    super_f[hh,ww,:] = r_set[nn][hh:hh+patch_size[0], ww:ww+patch_size[1], :].reshape(-1,)
                    
            super_f = super_f.reshape(-1, super_f.shape[2])
            super_f_norm = np.sqrt(np.sum(super_f**2, 1)).reshape(-1,1)
            super_f = super_f/super_f_norm
            
            dist = cdist(super_f, centers).reshape(h-patch_size[0]+1,w-patch_size[1]+1,-1)
            assert(dist.shape[2]==centers.shape[0])
            r_super_set[nn] = dist
            
            if nn%100 == 0:
                print(nn, end=' ')
            
            
        print('\n')
        
        with open(file_cache_feat, 'wb') as fh:
            pickle.dump([feat_set, r_set, r_super_set], fh)