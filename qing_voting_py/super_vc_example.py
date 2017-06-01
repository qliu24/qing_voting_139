import pickle
import numpy as np
from sklearn.cluster import KMeans
import time
from config_voting import *

file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}.pickle'.format('all', dataset_suffix, 'both'))
with open(file_cache_feat, 'rb') as fh:
    _,img_set,_ = pickle.load(fh)

print('total number of instances {0}'.format(len(img_set)))


fname = os.path.join(Feat['cache_dir'], 'dictionary_super_PASCAL3D+_VGG16_{0}_K{1}_tmp.pickle'.format(VC['layer'], VC['num_super']))
with open(fname, 'rb') as fh:
    patch_feature_dist, pos_rec = pickle.load(fh)


patch_size = [7,7]  # h, w of patch

assert(os.path.isfile(Dictionary_super))
with open(Dictionary_super, 'rb') as fh:
    assignment,centers = pickle.load(fh)
    
assert(VC['num_super'] == centers.shape[0])
        
print('total number of vectors {0}'.format(len(patch_feature_dist)))

patch_feature_dist_f = np.array([dd.reshape(-1,) for dd in patch_feature_dist])
patch_feature_dist_f_norm = np.sqrt(np.sum(patch_feature_dist_f**2, 1)).reshape(-1,1)
patch_feature_dist_f = patch_feature_dist_f/patch_feature_dist_f_norm
assert(patch_feature_dist_f.shape[1] == centers.shape[1])

K = VC['num_super']
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
        sys.stdout.flush()
        
save_path = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}_example.pickle'.format('all', dataset_suffix, 'both'))
with open(save_path, 'wb') as fh:
    pickle.dump(example, fh)