from joblib import Parallel, delayed
import pickle
import numpy as np
import math
from vcdist_funcs import *
import time

paral_num=20
# file_path = '/media/zzs/4TB/qingliu/qing_intermediate/feat_pickle/'
# file_path = '/mnt/4T-HD/qing/voting_data/feat_pickle/'
file_path = '/export/home/qliu24/qing_voting_data/intermediate/feat_pickle_alex/'
savename = file_path + 'all_simmat_mthrh048.pickle'
magic_thh = 0.48

layer_feature_dist = []
objects = ['car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train']
# objects = ['car']
for oo in objects:
    fname = file_path + 'res_info_' + oo + '_train_gray200.pickle'
    print('loading object {0}'.format(oo))
    with open(fname, 'rb') as fh:
        l, _, _ = pickle.load(fh)
        layer_feature_dist += l
        
        
N = len(layer_feature_dist)
print('total number of instances {0}'.format(N))

layer_feature_b = [None for nn in range(N)]
stride = 1
for nn in range(N):
    layer_feature_b[nn] = (layer_feature_dist[nn]<magic_thh).astype(int).T
    '''
    ihh, iww, idd = layer_feature_dist[nn].shape
    
    tmp = np.zeros((math.ceil(ihh/stride), math.ceil(iww/stride), idd)).astype('int')
    for hh in range(0,ihh,stride):
        for ww in range(0,iww,stride):
            d_min = np.unravel_index(np.argmin(layer_feature_dist[nn][hh:hh+stride,ww:ww+stride,:]), \
                                 layer_feature_dist[nn][hh:hh+stride,ww:ww+stride,:].shape)
            if np.min(layer_feature_dist[nn][hh:hh+stride,ww:ww+stride,:]) < magic_thh:
                tmp[int(hh/stride),int(ww/stride),d_min[2]] = 1
            
    layer_feature_b[nn] = tmp.T
    
    ihh, iww, idd = layer_feature_dist[nn].shape
    tmp = np.ones((math.ceil(ihh/stride), math.ceil(iww/stride), idd))
    for dd in range(idd):
        for hh in range(0,ihh,stride):
            for ww in range(0,iww,stride):
                min_dist = np.min(layer_feature_dist[nn][hh:hh+stride, ww:ww+stride, dd])
                tmp[int(hh/stride),int(ww/stride),dd] = min_dist
    
    layer_feature_b[nn] = (tmp<magic_thh).astype(int).T
    '''

print('Start compute sim matrix...', flush=True)
_s = time.time()

mat_dis1 = np.ones((N,N))
mat_dis2 = np.ones((N,N))
for nn1 in range(N-1):
    print(nn1, end=' ', flush=True)
    inputs = [(layer_feature_b[nn1], layer_feature_b[nn2]) for nn2 in range(nn1+1,N)]
    para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_sym2)(i) for i in inputs))
    mat_dis1[nn1, nn1+1:N] = para_rst[:,0]
    mat_dis2[nn1, nn1+1:N] = para_rst[:,1]

_e = time.time()
print((_e-_s)/60)
print(mat_dis1.shape)
print(mat_dis1[0])
print(mat_dis2.shape)
print(mat_dis2[0])

with open(savename, 'wb') as fh:
    pickle.dump([mat_dis1, mat_dis2], fh)

