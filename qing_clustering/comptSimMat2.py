from joblib import Parallel, delayed
import pickle
import numpy as np
import math
from vcdist_funcs import *
import time

paral_num=20
file_path = '/export/home/qliu24/VC_adv_data/cihang/adv_cls_patches/'
savename = file_path + 'simmat_vMFMM30_mthrh046.pickle'
magic_thh = 0.46

fname = file_path + 'pool4FeatVC.pickle'
with open(fname, 'rb') as fh:
    _, r_set_ori, _, r_set_fake = pickle.load(fh)
        
N = len(r_set_ori)
print('total number of instances {0}'.format(N))

layer_feature_b_ori = [None for nn in range(N)]
layer_feature_b_fake = [None for nn in range(N)]
for nn in range(N):
    layer_feature_b_ori[nn] = (r_set_ori[nn]<magic_thh).astype(int).T
    layer_feature_b_fake[nn] = (r_set_fake[nn]<magic_thh).astype(int).T

print('Start compute sim matrix ori...')
_s = time.time()

mat_dis1 = np.ones((N,N))
mat_dis2 = np.ones((N,N))
for nn1 in range(N-1):
    print(nn1, end=' ', flush=True)
    inputs = [(layer_feature_b_ori[nn1], layer_feature_b_ori[nn2]) for nn2 in range(nn1+1,N)]
    para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_sym2)(i) for i in inputs))
    mat_dis1[nn1, nn1+1:N] = para_rst[:,0]
    mat_dis2[nn1, nn1+1:N] = para_rst[:,1]

_e = time.time()
print((_e-_s)/60)
print(mat_dis1.shape)
print(mat_dis1[0])
print(mat_dis2.shape)
print(mat_dis2[0])

print('Start compute sim matrix fake...')
_s = time.time()

mat_dis1_fake = np.ones((N,N))
mat_dis2_fake = np.ones((N,N))
for nn1 in range(N-1):
    print(nn1, end=' ', flush=True)
    inputs = [(layer_feature_b_fake[nn1], layer_feature_b_fake[nn2]) for nn2 in range(nn1+1,N)]
    para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_sym2)(i) for i in inputs))
    mat_dis1_fake[nn1, nn1+1:N] = para_rst[:,0]
    mat_dis2_fake[nn1, nn1+1:N] = para_rst[:,1]

_e = time.time()
print((_e-_s)/60)
print(mat_dis1_fake.shape)
print(mat_dis1_fake[0])
print(mat_dis2_fake.shape)
print(mat_dis2_fake[0])

with open(savename, 'wb') as fh:
    pickle.dump([mat_dis1, mat_dis2, mat_dis1_fake, mat_dis2_fake], fh)


