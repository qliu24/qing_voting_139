import pickle
import numpy as np
from sklearn.cluster import KMeans
import time
from config_voting import *

n_cls=VC['num_super']

file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}.pickle'.format('all', dataset_suffix, 'both'))
with open(file_cache_feat, 'rb') as fh:
    _,_,r_set = pickle.load(fh)

print('total number of instances {0}'.format(len(r_set)))

patch_feature_dist = []
pos_rec = []
patch_size = [7,7]  # h, w of patch
for nn in range(len(r_set)):
    h,w = r_set[nn].shape[0:2]
    
    for hh in range(0, h-patch_size[0]+1, 3):
        for ww in range(0, w-patch_size[1]+1, 3):
            hh_t = min(hh+np.random.randint(3), h-patch_size[0])
            ww_t = min(ww+np.random.randint(3), w-patch_size[1])
            patch_feature_dist.append(r_set[nn][hh_t:hh_t+patch_size[0], ww_t:ww_t+patch_size[1], :])
            pos_rec.append([nn, hh_t, ww_t])
            
print('total number of vectors {0}'.format(len(patch_feature_dist)))

fname = os.path.join(Feat['cache_dir'], 'dictionary_super_PASCAL3D+_VGG16_{0}_K{1}_tmp.pickle'.format(VC['layer'], n_cls))
with open(fname, 'wb') as fh:
    pickle.dump([patch_feature_dist, pos_rec], fh)

patch_feature_dist_f = np.array([dd.reshape(-1,) for dd in patch_feature_dist])
patch_feature_dist_f_norm = np.sqrt(np.sum(patch_feature_dist_f**2, 1)).reshape(-1,1)
patch_feature_dist_f = patch_feature_dist_f/patch_feature_dist_f_norm

print('start Kmeans')

_s = time.time()
km = KMeans(n_clusters=n_cls, init='k-means++', random_state=99, n_jobs=-5)
assignment = km.fit_predict(patch_feature_dist_f)
centers = km.cluster_centers_
_e = time.time()
print('Kmeans time: {0}'.format((_e-_s)/60))

fname = os.path.join(Feat['cache_dir'], 'dictionary_super_PASCAL3D+_VGG16_{0}_K{1}.pickle'.format(VC['layer'], n_cls))
with open(fname, 'wb') as fh:
    pickle.dump([assignment, centers], fh)