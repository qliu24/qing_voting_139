import pickle
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist

category='aeroplane'
K = 4
magic_thh = 0.47
file_path = '/export/home/qliu24/VC_adv_data/qing/VGG_adv/feat/'
sim_fname = file_path + 'simmat_{}_mthrh047_allVC.pickle'.format(category)
savename = file_path + 'mix_model/{}_K{}_notrain.pickle'.format(category, K)
feat_fname = file_path + 'pool4FeatVC_{}_train.pickle'.format(category)

dict_file='/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_all_VGG16_pool4_K200_vMFMM30.pickle'
# dict_file='/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_car_VGG16_pool4_K200_vMFMM30.pickle'

# Spectral clustering based on the similarity matrix
with open(sim_fname, 'rb') as fh:
    mat_dis1, _ = pickle.load(fh)

mat_dis = mat_dis1
N = mat_dis.shape[0]
print('total number of instances {0}'.format(N))

mat_full = mat_dis + mat_dis.T - np.ones((N,N))
np.fill_diagonal(mat_full, 0)

W_mat = 1. - mat_full
print('W_mat stats: {}, {}'.format(np.mean(W_mat), np.std(W_mat)))

K1 = 2
cls_solver = SpectralClustering(n_clusters=K1,affinity='precomputed', random_state=666)
lb = cls_solver.fit_predict(W_mat)

K2=2
idx2 = []
W_mat2 = []
lb2 = []
for k in range(K1):
    idx2.append(np.where(lb==k)[0])
    W_mat2.append(W_mat[np.ix_(idx2[k],idx2[k])])
    print('W_mat_i stats: {}, {}'.format(np.mean(W_mat2[k]), np.std(W_mat2[k])))
    
    cls_solver = SpectralClustering(n_clusters=K2,affinity='precomputed', random_state=999)
    lb2.append(cls_solver.fit_predict(W_mat2[k]))
    
rst_lbs1 = np.ones(len(idx2[0]))*-1
rst_lbs1[np.where(lb2[0]==0)[0]] = 0
rst_lbs1[np.where(lb2[0]==1)[0]] = 1
rst_lbs2 = np.ones(len(idx2[1]))*-1
rst_lbs2[np.where(lb2[1]==0)[0]] = 2
rst_lbs2[np.where(lb2[1]==1)[0]] = 3
rst_lbs = np.ones(N)*-1
rst_lbs[idx2[0]] = rst_lbs1
rst_lbs[idx2[1]] = rst_lbs2
rst_lbs = rst_lbs.astype('int')

# Load the feature vector and compute VC encoding
with open(feat_fname, 'rb') as fh:
    layer_feature = pickle.load(fh)
    
with open(dict_file, 'rb') as fh:
    _, centers, _ = pickle.load(fh)


assert(N == len(layer_feature))

r_set = [None for nn in range(N)]
for nn in range(N):
    iheight,iwidth = layer_feature[nn].shape[0:2]
    lff = layer_feature[nn].reshape(-1, 512)
    lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
    r_set[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
    
layer_feature_b = [None for nn in range(N)]
for nn in range(N):
    layer_feature_b[nn] = (r_set[nn]<magic_thh).astype(int).T
    
    
# Compute the unary weights
# VC num
max_0 = max([layer_feature_b[nn].shape[0] for nn in range(N)])
# width
max_1 = max([layer_feature_b[nn].shape[1] for nn in range(N)])
# height
max_2 = max([layer_feature_b[nn].shape[2] for nn in range(N)])
print(max_0, max_1, max_2)

all_train = [np.zeros((max_0, max_1, max_2)) for kk in range(K)]
all_N = [0 for kk in range(K)]
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
    ki = rst_lbs[nn]
    all_train[ki] += padded
    all_N[ki] += 1
    
assert(N == np.sum(all_N))
all_weights = [None for kk in range(K)]
for kk in range(K):
    probs = all_train[kk]/all_N[kk] + 1e-3
    all_weights[kk] = np.log(probs/(1.-probs))
    
all_priors = np.array(all_N)/N

with open(savename, 'wb') as fh:
    pickle.dump([all_weights, all_priors], fh)