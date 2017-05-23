from joblib import Parallel, delayed
import pickle
import numpy as np
from vcdist_funcs import *
import time

paral_num=8
# file_path = '/media/zzs/4TB/qingliu/qing_intermediate/feat_pickle/'
# file_path = '/mnt/4T-HD/qing/voting_data/feat_pickle/'
file_path = '/mnt/4T-HD/qing/intermediate/feat_pickle/'
savename = file_path + 'all_super_simmat.pickle'
# magic_thh = 0.83

layer_feature_dist = []
sub_type = []
view_point = []
objects = ['car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train']
for oo in objects:
    fname = file_path + 'res_info_' + oo + '_train.pickle'
    print('loading object {0}'.format(oo))
    with open(fname, 'rb') as fh:
        _, l, s, v = pickle.load(fh)
        layer_feature_dist += l
        sub_type += s
        view_point += v
        
N = len(sub_type)
print('total number of instances {0}'.format(N))

layer_feature_b = [None for nn in range(N)]
for nn in range(N):
    # layer_feature_b[nn] = (layer_feature_dist[nn]<magic_thh).astype(int).T
    tmp = np.zeros_like(layer_feature_dist[nn]).astype('int')
    for hh in range(tmp.shape[0]):
        for ww in range(tmp.shape[1]):
            tmp[hh,ww,np.argmin(tmp[hh,ww])] = 1
            
    layer_feature_b[nn] = tmp.T

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

