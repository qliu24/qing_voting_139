from joblib import Parallel, delayed
import pickle
import numpy as np
import math
from scipy.spatial.distance import cdist
from vcdist_funcs import *
import time

category = 'aeroplane'
paral_num = 30
file_path = '/export/home/qliu24/VC_adv_data/qing/VGG_adv/feat/'
savename = file_path + 'simmat_{}_mthrh047_allVC.pickle'.format(category)
magic_thh = 0.47

fname = file_path + 'pool4FeatVC_{}.pickle'.format(category)
with open(fname, 'rb') as fh:
    layer_feature = pickle.load(fh)
    
dict_file='/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_all_VGG16_pool4_K200_vMFMM30.pickle'
with open(dict_file, 'rb') as fh:
    _, centers, _ = pickle.load(fh)


N = len(layer_feature)
print('total number of instances {0}'.format(N))

r_set = [None for nn in range(N)]
# r_set_fake = [None for nn in range(N)]
for nn in range(N):
    iheight,iwidth = layer_feature[nn].shape[0:2]
    lff = layer_feature[nn].reshape(-1, 512)
    lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
    r_set[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
    '''
    iheight,iwidth = layer_feature_fake[nn].shape[0:2]
    lff = layer_feature_fake[nn].reshape(-1, 512)
    lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
    r_set_fake[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
    '''

layer_feature_b = [None for nn in range(N)]
# layer_feature_b_fake = [None for nn in range(N)]
for nn in range(N):
    layer_feature_b[nn] = (r_set[nn]<magic_thh).astype(int).T
    # layer_feature_b_fake[nn] = (r_set_fake[nn]<magic_thh).astype(int).T

print('Start compute sim matrix ori...')
_s = time.time()

inputs = [(layer_feature_b, nn) for nn in range(N)]
para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral)(i) for i in inputs))
mat_dis1 = para_rst[:,0]
mat_dis2 = para_rst[:,1]

_e = time.time()
print((_e-_s)/60)
print(mat_dis1.shape)
print(mat_dis1[0])
print(mat_dis2.shape)
print(mat_dis2[0])

with open(savename, 'wb') as fh:
    pickle.dump([mat_dis1, mat_dis2], fh)

'''
print('Start compute sim matrix fake...')
_s = time.time()

inputs = [(layer_feature_b_fake, nn) for nn in range(N)]
para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral)(i) for i in inputs))
mat_dis1_fake = para_rst[:,0]
mat_dis2_fake = para_rst[:,1]

_e = time.time()
print((_e-_s)/60)
print(mat_dis1_fake.shape)
print(mat_dis1_fake[0])
print(mat_dis2_fake.shape)
print(mat_dis2_fake[0])

with open(savename, 'wb') as fh:
    pickle.dump([mat_dis1, mat_dis2, mat_dis1_fake, mat_dis2_fake], fh)


'''