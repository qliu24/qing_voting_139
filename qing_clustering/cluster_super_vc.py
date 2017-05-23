import pickle
import numpy as np
from sklearn.cluster import KMeans
import time

file_path = '/mnt/4T-HD/qing/intermediate/feat_pickle/'
magic_thh = 0.83
N = 300

layer_feature_dist = []
sub_type = []
view_point = []
objs = ['car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train']
# objs = ['car']
for oo in objs:
    fname = file_path + 'res_info_' + oo + '_train.pickle'
    print('loading object {0}'.format(oo))
    with open(fname, 'rb') as fh:
        l, s, v = pickle.load(fh)
        Nidx = np.random.choice(len(s), size=(N,), replace=False)
        layer_feature_dist += [l[nii].T for nii in Nidx]
        sub_type += [s[nii] for nii in Nidx]
        view_point += [v[nii] for nii in Nidx]
        
print('total number of instances {0}'.format(len(sub_type)))

patch_feature_dist = []
pos_rec = []
patch_size = [7,7]  # h, w of patch
for nn in range(N*len(objs)):
    w,h = layer_feature_dist[nn].shape[1:]
    
    for ww in range(0, w-patch_size[1]+1, 3):
        for hh in range(0, h-patch_size[0]+1, 3):
            patch_feature_dist.append(layer_feature_dist[nn][:,ww:ww+patch_size[1], hh:hh+patch_size[0]])
            pos_rec.append([nn, ww, hh])
            
            
print('total number of vectors {0}'.format(len(patch_feature_dist)))

patch_feature_dist_f = np.array([dd.reshape(-1,) for dd in patch_feature_dist])
patch_feature_dist_f_norm = np.sqrt(np.sum(patch_feature_dist_f**2, 1)).reshape(-1,1)
patch_feature_dist_f = patch_feature_dist_f/patch_feature_dist_f_norm

print('start Kmeans')
n_cls=80
_s = time.time()
km = KMeans(n_clusters=n_cls, init='k-means++', random_state=99, n_jobs=-5)
assignment = km.fit_predict(patch_feature_dist_f)
centers = km.cluster_centers_
_e = time.time()
print('Kmeans time: {0}'.format((_e-_s)/60))

with open('/home/candy/qing_voting_139/qing_clustering/super_vc_all_K80.py', 'wb') as fh:
    pickle.dump([assignment, centers], fh)
