import pickle
import numpy as np
from scipy.spatial.distance import cdist

def get_stats(oo, magic_thh = 0.83, if_bg=False):
    file_path = '/export/home/qliu24/qing_voting_data/intermediate/feat_VGG/'
    filename = file_path + '{}_mergelist_rand_train.pickle'.format(oo)
    # file_path = '/export/home/qliu24/VC_adv_data/qing/VGG_adv/feat/'
    # filename = file_path + 'pool4FeatVC_{}.pickle'.format(oo)
    if if_bg:
        filename = file_path + 'pool4FeatVC_{}_bg.pickle'.format(oo)
        
    
    print(filename)
    '''
    with open(filename, 'rb') as fh:
        _, layer_feature_dist = pickle.load(fh)
    
    '''
    with open(filename, 'rb') as fh:
        layer_feature,_,_ = pickle.load(fh)
        
    dict_file='/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_all_VGG16_pool4_K200_vMFMM30.pickle'
    with open(dict_file, 'rb') as fh:
        _, centers, _ = pickle.load(fh)
        
    # centers = centers/np.sqrt(np.sum(centers**2, axis=1)).reshape(-1,1)
    
    layer_feature_dist = []
    for nn in range(len(layer_feature)):
        iheight,iwidth = layer_feature[nn].shape[0:2]
        lff = layer_feature[nn].reshape(-1, 512)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        layer_feature_dist.append(cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1))
        
    
    N = len(layer_feature_dist)
    print('{0}: total number of instances {1}'.format(oo, N))
    print(layer_feature_dist[0].shape)
    
    
    layer_feature_b = [None for nn in range(N)]
    vc_cnt = [None for nn in range(N)]
    for nn in range(N):
        layer_feature_b[nn] = (layer_feature_dist[nn]<magic_thh).astype(int).T
        vc_cnt[nn] = np.sum(np.sum(layer_feature_b[nn], axis=1), axis=1)
        
    vc_cnt = np.array(vc_cnt)
    vc_avg = np.mean(vc_cnt, axis=0)
    
    vc_cnt2 = [None for nn in range(N)]
    vc_stat1 = [None for nn in range(N)]
    vc_stat2 = [None for nn in range(N)]
    for nn in range(N):
        vc_cnt2[nn] = np.mean(np.sum(layer_feature_b[nn], axis=0))
        vc_stat1[nn] = np.sum(np.sum(layer_feature_b[nn], axis=0)==0)/(layer_feature_b[nn].shape[1]*layer_feature_b[nn].shape[2])
        vc_stat2[nn] = np.mean(np.min(layer_feature_dist[nn].T, axis=0))
    
    # print('{0} avgnum of VC fired at each pixel:'.format(oo))
    print('{0}, {1}'.format(np.mean(vc_cnt2), np.mean(vc_stat1)))
    # print(np.mean(vc_stat2))
    # print(np.max(layer_feature_dist_super[0]), np.min(layer_feature_dist_super[0]))
    return vc_avg



# objects = ['car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train']
objects = ['car', 'aeroplane', 'bus']
for oo in objects:
    vc_avg = get_stats(oo, magic_thh = 0.47)
