from sklearn.cluster import KMeans
from config_voting import *

cluster_num = 200
save_path = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_ALEX_{0}_K{1}_gray.pickle'.format(VC['layer'], cluster_num)


# number of files to read in
file_num = 6
feat_set = np.zeros((featDim, 0))
for ii in range(file_num):
    print('loading file {0}/{1}'.format(ii+1, file_num))
    fname = Dict['cache_path']+str(ii)+'_gray.pickle'
    with open(fname, 'rb') as fh:
        res, _, _ = pickle.load(fh)
        feat_set = np.column_stack((feat_set, res))
        

print('all feat_set')
print(feat_set.shape)

# L2 normalization as preprocessing
feat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/feat_norm

print('Start K-means...')
_s = time.time()
km = KMeans(n_clusters=cluster_num, init='k-means++', random_state=99, n_jobs=-1)
assignment = km.fit_predict(feat_set.T)
centers = km.cluster_centers_
_e = time.time()
print('K-means running time: {0}'.format((_e-_s)/60))

with open(save_path, 'wb') as fh:
    pickle.dump([assignment, centers], fh)
