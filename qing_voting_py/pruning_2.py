import pickle
import numpy as np
from config_voting import *

cluster_num = featDim
cluster_file = '/home/candy/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_VGG16_pool4_K189_prune_512.pickle'

print('loading data...')

# number of files to read in
file_num = 6
feat_set = np.zeros((featDim, 0))
loc_set = np.zeros((5, 0), dtype='int')
img_set = []
for ii in range(file_num):
    print('loading file {0}/{1}'.format(ii+1, file_num))
    fname = Dict['cache_path']+str(ii)+'.pickle'
    with open(fname, 'rb') as fh:
        res, iloc, iimg = pickle.load(fh)
        feat_set = np.column_stack((feat_set, res))
        loc_set = np.column_stack((loc_set, iloc.astype('int')))
        img_set += iimg
        
print('all feat_set')
print(feat_set.shape)
print('all loc_set')
print(loc_set.shape)
print('all images')
print(len(img_set))

with open(cluster_file, 'rb') as fh:
    assignment, centers = pickle.load(fh)
    
K = centers.shape[0]

feat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/feat_norm

# the num of images for each cluster
num = 50
print('save top {0} images for each cluster'.format(num))
example = [None for nn in range(K)]
for k in range(K):
    target = centers[k]
    index = np.where(assignment == k)[0]
    num = min(num, len(index))
    
    tempFeat = feat_set[:,index]
    error = np.sum((tempFeat.T - target)**2, 1)
    sort_idx = np.argsort(error)
    patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
    for idx in range(num):
        iloc = loc_set[:,index[sort_idx[idx]]]
        patch = img_set[iloc[0]][iloc[1]:iloc[3], iloc[2]:iloc[4], :]
        patch_set[:,idx] = patch.flatten()
        
    example[k] = np.copy(patch_set)
    if k%10 == 0:
        print(k)
        
save_path = '/home/candy/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_VGG16_pool4_K189_prune_512_example.pickle'
with open(save_path, 'wb') as fh:
    pickle.dump(example, fh)
