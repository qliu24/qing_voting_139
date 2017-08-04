import numpy as np
from vMFMM import *
import pickle

cluster_num = 512
file_num = 7
featDim = 512
Arf = 100
feat_set = np.zeros((featDim, 0))
loc_set = np.zeros((5, 0), dtype='int')
for ii in range(file_num):
    print('loading file {0}'.format(ii))
    fname = '/export/home/qliu24/ILSVRC12_VC/feat/pool4_all_dumped_data{}.pickle'.format(ii)
    with open(fname, 'rb') as fh:
        res, iloc = pickle.load(fh)
        feat_set = np.column_stack((feat_set, res))
        loc_set = np.column_stack((loc_set, iloc.astype('int')))

print('all feat_set')
feat_set = feat_set.T
print(feat_set.shape)

model = vMFMM(cluster_num,'k++')
model.fit(feat_set, 30, max_it=150)

save_path = '/export/home/qliu24/ILSVRC12_VC/dictionary/dictionary_ILSVRC12_VGG16_pool4_K512_vMFMM30.pickle'
with open(save_path, 'wb') as fh:
    pickle.dump([model.p, model.mu, model.pi], fh)

############## save examples ###################

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