import pickle
import numpy as np
from sklearn.cluster import KMeans
import time
from config_voting import *

file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}.pickle'.format('all', dataset_suffix, 'both'))
with open(file_cache_feat, 'rb') as fh:
    _,img_set,r_set = pickle.load(fh)

print('total number of instances {0}'.format(len(r_set)))

patch_feature_dist = []
pos_rec = []
patch_size = [7,7]  # h, w of patch
for nn in range(len(r_set)):
    h,w = r_set[nn].shape[0:2]
    
    for hh in range(0, h-patch_size[0]+1, 3):
        for ww in range(0, w-patch_size[1]+1, 3):
            patch_feature_dist.append(r_set[nn][hh:hh+patch_size[0], ww:ww+patch_size[1], :])
            pos_rec.append([nn, hh, ww])
            
assert(os.path.isfile(Dictionary_super))
with open(Dictionary_super, 'rb') as fh:
    assignment,centers = pickle.load(fh)
        
print('total number of vectors {0}'.format(len(patch_feature_dist)))

patch_feature_dist_f = np.array([dd.reshape(-1,) for dd in patch_feature_dist])
patch_feature_dist_f_norm = np.sqrt(np.sum(patch_feature_dist_f**2, 1)).reshape(-1,1)
patch_feature_dist_f = patch_feature_dist_f/patch_feature_dist_f_norm
assert(patch_feature_dist_f.shape[1] == centers.shape[1])

K = centers.shape[0]
example = [None for kk in range(K)]
num = 50
for kk in range(K):
    target = centers[kk]
    index = np.where(assignment == kk)[0]
    num = min(num, len(index))
    
    tempFeat = patch_feature_dist_f[index,:]
    error = np.sum((tempFeat - target)**2, 1)
    sort_idx = np.argsort(error)
    patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
    for idx in range(num):
        nn, hh, ww = pos_rec[index[sort_idx[idx]]]
        hi = Astride * (hh + patch_size[0]//2) - Apad
        wi = Astride * (ww + patch_size[1]//2) - Apad
        
        patch = img_set[nn][hi:hi+Arf, wi:wi+Arf, :]
        patch_set[:,idx] = patch.flatten()
        
    example[kk] = np.copy(patch_set)
    if kk%10 == 0:
        print(kk)
        
save_path = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}_example.pickle'.format('all', dataset_suffix, 'both'))
with open(save_path, 'wb') as fh:
    pickle.dump(example, fh)