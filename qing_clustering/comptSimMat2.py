from joblib import Parallel, delayed
import pickle
import numpy as np
import math
from scipy.spatial.distance import cdist
from vcdist_funcs import *
import time

paral_num=30
file_path = '/export/home/qliu24/VC_adv_data/cihang/adv_cls_patches/'
savename = file_path + 'simmat_carplane_vMFMM30_mthrh047_allVC.pickle'
magic_thh = 0.47

fname = file_path + 'pool4FeatVC_car.pickle'
with open(fname, 'rb') as fh:
    layer_feature_ori, _, layer_feature_fake, _ = pickle.load(fh)

fname = file_path + 'pool4FeatVC_aeroplane.pickle'
with open(fname, 'rb') as fh:
    layer_feature_ap = pickle.load(fh)
    
dict_file='/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_all_VGG16_pool4_K200_vMFMM30.pickle'
with open(dict_file, 'rb') as fh:
    _, centers, _ = pickle.load(fh)

layer_feature_ori = layer_feature_ori[0:1000] + layer_feature_ap
layer_feature_fake = layer_feature_fake[0:1000] + layer_feature_ap

N = len(layer_feature_ori)
print('total number of instances {0}'.format(N))

r_set_ori = [None for nn in range(N)]
r_set_fake = [None for nn in range(N)]
for nn in range(N):
    iheight,iwidth = layer_feature_ori[nn].shape[0:2]
    lff = layer_feature_ori[nn].reshape(-1, 512)
    lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
    r_set_ori[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
    
    iheight,iwidth = layer_feature_fake[nn].shape[0:2]
    lff = layer_feature_fake[nn].reshape(-1, 512)
    lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
    r_set_fake[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)

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


