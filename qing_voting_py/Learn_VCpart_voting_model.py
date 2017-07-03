import pickle
import numpy as np
from scipy.spatial.distance import pdist
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
    save_name = file_path2+'VCpart_model_{0}_K{1}.pickle'.format(oo, K)
else:
    lbs = np.zeros(N)
    K=1
    save_name = file_path2+'VCpart_model_{0}_bg.pickle'.format(oo)
    

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
    
    
    heat_map_init = [[] for vc_i in range(max_2)]
    for vc_i in range(max_2):
        vc_p = []
        for nn in range(N):
            vc_l = layer_feature_b[nn][:,:,vc_i]
            rll, cll = vc_l.shape
            diffr = int((max_0-rll)/2)
            diffc = int((max_1-cll)/2)
            row_i, col_i = np.where(vc_l==1)
            for pp in zip(row_i,col_i):
                vc_p.append((pp[0]+diffr,pp[1]+diffc))

        if len(vc_p)<N/2:
            continue

        vc_p=np.array(vc_p)
        clnum=4
        gm, assignment = gm_vc_pos(vc_p, clnum)
        while True:
            p_dist = pdist(gm.means_)
            if np.any(gm.weights_<0.5/clnum) or np.any(p_dist<9):
                clnum -= 1
                if clnum==1:
                    gm = None
                    assignment = np.zeros(vc_p.shape[0])
                    break
                else:
                    gm, assignment = gm_vc_pos(vc_p, clnum)
            else:
                break

        # print('final K: {0}'.format(K), flush='True')

        for kcc in range(clnum):
            vc_p_kk = vc_p[assignment==kcc]
            if len(vc_p_kk)<10:
                print('cluster too small {0}_{1}'.format(vc_i, kcc))
                continue

            heat_map_init[vc_i].append(init_heatmap(vc_p_kk, max_0, max_1, 10, False))

        if vc_i%20==0:
            print(vc_i, end=' ', flush=True)
            
    print('')
    
    layer_fired_pos, sc, sc_d = get_fired_pos(layer_feature_dist_kk, layer_feature_b, heat_map_init)
    print('iter {0} score: {1}, {2}'.format(-1, sc, sc_d))

    for itt in range(15):
        vc_templates = get_vctplt(layer_fired_pos, layer_feature_dist_kk, max_0, max_1)
        heat_map = get_heatmap(layer_fired_pos, max_0, max_1, max_2)

        layer_fired_pos, sc, sc_d = get_fired_pos(layer_feature_dist_kk, layer_feature_b, heat_map, vc_templates)
        print('iter {0} score: {1}, {2}'.format(itt, sc, sc_d))
    
    
    hm_ls.append(heat_map)
    vcT_ls.append(vc_templates)
    
with open(save_name, 'wb') as fh:
    pickle.dump([hm_ls, vcT_ls], fh)
    
        


