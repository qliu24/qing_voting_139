import pickle
import numpy as np
from copy import *
from VCpart_model_func import *

oo='car'
if_bg = False
magic_thh = 0.85

file_path1 = '/export/home/qliu24/qing_voting_data/intermediate/feat_car/'
file_path2 = '/export/home/qliu24/qing_voting_data/intermediate/VCpart_model_car/'

if not if_bg:
    fname = file_path1 + oo + '_mergelist_rand_train_carVC.pickle'
    print('loading object {0}'.format(oo))
    with open(fname, 'rb') as fh:
        _,layer_feature_dist = pickle.load(fh)
else:
    fname = file_path1 + oo + '_mergelist_rand_train_carVC_bg.pickle'
    print('loading object {0}'.format(oo))
    with open(fname, 'rb') as fh:
        _,layer_feature_dist = pickle.load(fh)
            
N = len(layer_feature_dist)
print('total number of instances {0}'.format(N))

if not if_bg:
    fname = file_path2 + '{0}_k2_2_lbs.pickle'.format(oo)
    with open(fname, 'rb') as fh:
        lbs = pickle.load(fh)
        
    K=4
    save_name = file_path2+'VCpart_model_{0}_K{1}_pctl33.pickle'.format(oo, K)
else:
    lbs = np.zeros(N)
    K=1
    save_name = file_path2+'VCpart_model_{0}_bg_pctl33.pickle'.format(oo)
    

hm_ls = []
vcT_ls = []
for kk in range(K):
    idx_s = np.where(lbs==kk)[0]
    
    layer_feature_dist_kk = [layer_feature_dist[nn] for nn in idx_s]
    
    N = len(layer_feature_dist_kk)
    print('total number of instances {0} for cluster {1}'.format(N, kk))
    
    
    layer_feature_b = [None for nn in range(N)]
    for nn in range(N):
        lfb = (layer_feature_dist_kk[nn]<magic_thh).astype(int)
        lfb[0:3, :, :] = 0
        lfb[-3:, :, :] = 0
        lfb[:, 0:3, :] = 0
        lfb[:, -3:, :] = 0
        layer_feature_b[nn] = deepcopy(lfb)
    
    max_0 = max([lfb.shape[0] for lfb in layer_feature_b])
    max_1 = max([lfb.shape[1] for lfb in layer_feature_b])
    max_2 = max([lfb.shape[2] for lfb in layer_feature_b])
    
    print('heatmap shape: {0}'.format([max_0, max_1, max_2]))
    
    heat_map = get_blur_heatmap(layer_feature_b, max_0, max_1, max_2)
    heat_map = trim_heatmap(heat_map, 50)
    heat_map = normalize_heatmap(heat_map)
    
    layer_feature_nms, sc, sc_d = get_nms(layer_feature_dist_kk, layer_feature_b, heat_map)
    print('iter {0} score: {1}, {2}'.format(-1, sc, sc_d))
    
    for itt in range(15):
        vc_templates = get_vctplt(layer_feature_nms, layer_feature_dist_kk)
        heat_map = get_blur_heatmap(layer_feature_nms, max_0, max_1, max_2)
        heat_map = trim_heatmap(heat_map, 33)
        heat_map = normalize_heatmap(heat_map)
        
        layer_feature_nms, sc, sc_d = get_nms(layer_feature_dist_kk, layer_feature_b, heat_map, vc_templates)
        print('iter {0} score: {1}, {2}'.format(itt, sc, sc_d))
        
    hm_ls.append(heat_map)
    vcT_ls.append(vc_templates)
    
with open(save_name, 'wb') as fh:
    pickle.dump([hm_ls, vcT_ls], fh)
    
        

