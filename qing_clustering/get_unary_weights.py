import h5py
import numpy as np
import pickle
import glob
from scipy.stats import spearmanr
import math

def get_unary_weights(category, magic_thrh = 0.046, stride=2, if_bg = True):
    file_path = '/mnt/4T-HD/qing/intermediate/feat/'
    filename = file_path + '{0}_mergelist_rand_train.pickle'.format(category)
    if if_bg:
        filename = file_path + '{0}_mergelist_rand_train_bg.pickle'.format(category)
        
    print(filename)
    with open(filename, 'rb') as fh:
        _, _, r_super_set = pickle.load(fh)
    
    N = len(r_super_set)
    layer_feature_b = [None for nn in range(N)]
    
    for nn in range(N):
        # layer_feature_b[nn] = (layer_feature_dist[nn]<magic_thh).astype(int)
        
        ihh, iww, idd = r_super_set[nn].shape
        '''
        tmp = np.zeros((math.ceil(ihh/stride), math.ceil(iww/stride), idd)).astype('int')
        for hh in range(0,ihh,stride):
            for ww in range(0,iww,stride):
                d_min = np.unravel_index(np.argmin(r_super_set[nn][hh:hh+stride,ww:ww+stride,:]), \
                                         r_super_set[nn][hh:hh+stride,ww:ww+stride,:].shape)
                if np.min(r_super_set[nn][hh:hh+stride,ww:ww+stride,:]) < magic_thrh:
                    tmp[int(hh/stride),int(ww/stride),d_min[2]] = 1
                
        layer_feature_b[nn] = tmp.T
        '''
        tmp = np.ones((math.ceil(ihh/stride), math.ceil(iww/stride), idd))
        for dd in range(idd):
            for hh in range(0,ihh,stride):
                for ww in range(0,iww,stride):
                    min_dist = np.min(r_super_set[nn][hh:hh+stride, ww:ww+stride, dd])
                    tmp[int(hh/stride),int(ww/stride),dd] = min_dist
        
        layer_feature_b[nn] = (tmp<magic_thrh).astype(int).T
        
    max_0 = max([layer_feature_b[nn].shape[0] for nn in range(N)])
    max_1 = max([layer_feature_b[nn].shape[1] for nn in range(N)])
    max_2 = max([layer_feature_b[nn].shape[2] for nn in range(N)])
    all_bg_b = np.zeros((max_0, max_1, max_2))
    for nn in range(N):
        vnum, ww, hh = layer_feature_b[nn].shape
        assert(vnum == max_0)
        diff_w1 = int((max_1-ww)/2)
        diff_w2 = int(max_1-ww-diff_w1)
        assert(max_1 == diff_w1+diff_w2+ww)
        
        diff_h1 = int((max_2-hh)/2)
        diff_h2 = int(max_2-hh-diff_h1)
        assert(max_2 == diff_h1+diff_h2+hh)
        
        padded = np.pad(layer_feature_b[nn], ((0,0),(diff_w1, diff_w2),(diff_h1, diff_h2)), 'constant', constant_values=0)
        all_bg_b += np.pad(layer_feature_b[nn], ((0,0),(diff_w1, diff_w2),(diff_h1, diff_h2)), 'constant', constant_values=0)
        
    probs = all_bg_b/N + 1e-3
    
    weights = np.log(probs/(1.-probs))
    save_path = '/mnt/4T-HD/qing/intermediate/unary_weights/'
    savefile = save_path+'{0}_train.pickle'.format(category)
    if if_bg:
        savefile = save_path+'{0}_train_bg.pickle'.format(category)
        
    with open(savefile, 'wb') as fh:
        pickle.dump(weights, fh)