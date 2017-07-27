import numpy as np
from vMFMM import *
import pickle

cluster_num = 200
file_num = [0,1,2,4,5]
featDim = 512
Arf = 100
feat_set = np.zeros((featDim, 0))
loc_set = np.zeros((5, 0), dtype='int')
img_set = []
for ii in file_num:
    print('loading file {0}'.format(ii))
    fname = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/pool4_all_dumped_data'+str(ii)+'.pickle'
    with open(fname, 'rb') as fh:
        res, iloc, iimg = pickle.load(fh)
        feat_set = np.column_stack((feat_set, res))
        loc_set = np.column_stack((loc_set, iloc.astype('int')))
        img_set += iimg

print('all feat_set')
feat_set = feat_set.T
print(feat_set.shape)

model = vMFMM(cluster_num,'k++')
model.fit(feat_set, 30, max_it=200)

save_path = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_nobus_VGG16_pool4_K200_vMFMM30.pickle'
with open(save_path, 'wb') as fh:
    pickle.dump([model.p, model.mu, model.pi], fh)

num = 50
print('save top {0} images for each cluster'.format(num))
example = [None for vc_i in range(cluster_num)]
for vc_i in range(cluster_num):
    patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
    sort_idx = np.argsort(-model.p[:,vc_i])[0:num]
    for idx in range(num):
        iloc = loc_set[:,sort_idx[idx]]
        patch = img_set[iloc[0]][iloc[1]:iloc[3], iloc[2]:iloc[4], :]
        patch_set[:,idx] = patch.flatten()
        
    example[vc_i] = np.copy(patch_set)
    if vc_i%10 == 0:
        print(vc_i)
        
save_path2 = save_path.replace('.pickle','_example.pickle')
with open(save_path2, 'wb') as fh:
    pickle.dump(example, fh)