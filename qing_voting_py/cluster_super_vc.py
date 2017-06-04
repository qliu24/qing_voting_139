from sklearn.cluster import KMeans
import time
from scipy.spatial.distance import cdist
from config_voting import *
from copy import *

n_cls=VC['num_super']

file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}.pickle'.format('all', dataset_suffix, 'both'))
with open(file_cache_feat, 'rb') as fh:
    feat_set,_,r_set = pickle.load(fh)

print('total number of instances {0}'.format(len(r_set)))

assert(os.path.isfile(Dictionary_car))
with open(Dictionary_car, 'rb') as fh:
    _,centers_vc = pickle.load(fh)

patch_feature_dist = []
pos_rec = []
r_set_new = [None for nn in range(1000)]
# for nn in range(len(r_set)):
for nn in range(1000):
    height, width = feat_set[nn].shape[0:2]
    assert(featDim == feat_set[nn].shape[2])
    
    layer_feature = feat_set[nn].reshape(-1, featDim)
    feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
    layer_feature = layer_feature/feat_norm
    
    dist = cdist(layer_feature, centers_vc).reshape(height,width,-1)
    assert(dist.shape[2]==centers_vc.shape[0]);
    r_set_new[nn] = deepcopy(dist)
    
    h,w = r_set_new[nn].shape[0:2]
    
    for hh in range(0, h-patch_size[0]+1, 3):
        for ww in range(0, w-patch_size[1]+1, 3):
            hh_t = min(hh+np.random.randint(3), h-patch_size[0])
            ww_t = min(ww+np.random.randint(3), w-patch_size[1])
            patch_feature_dist.append(r_set_new[nn][hh_t:hh_t+patch_size[0], ww_t:ww_t+patch_size[1], :])
            pos_rec.append([nn, hh_t, ww_t])
            
print('total number of vectors {0}'.format(len(patch_feature_dist)))

fname = os.path.join(Feat['cache_dir'], 'dictionary_super_PASCAL3D+_car_VGG16_{0}_K{1}_tmp.pickle'.format(VC['layer'], n_cls))
with open(fname, 'wb') as fh:
    pickle.dump([patch_feature_dist, pos_rec], fh)

patch_feature_dist_f = np.array([dd.reshape(-1,) for dd in patch_feature_dist])
patch_feature_dist_f_norm = np.sqrt(np.sum(patch_feature_dist_f**2, 1)).reshape(-1,1)
patch_feature_dist_f = patch_feature_dist_f/patch_feature_dist_f_norm

print('start Kmeans')

_s = time.time()
km = KMeans(n_clusters=n_cls, init='k-means++', random_state=99, n_jobs=1)
assignment = km.fit_predict(patch_feature_dist_f)
centers = km.cluster_centers_
_e = time.time()
print('Kmeans time: {0}'.format((_e-_s)/60))

fname = os.path.join(Feat['cache_dir'], 'dictionary_super_PASCAL3D+_car_VGG16_{0}_K{1}.pickle'.format(VC['layer'], n_cls))
with open(fname, 'wb') as fh:
    pickle.dump([assignment, centers], fh)