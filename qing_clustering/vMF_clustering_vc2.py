import numpy as np
from vMFMM import *
import pickle

cluster_num = 100
featDim = 128
Arf = 16

all_categories = ['car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train']

for ii,cc in enumerate(all_categories):
    # ii += 2
    print('loading file {0}'.format(ii))
    fname = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/pool2_all_dumped_data{}.pickle'.format(ii)
    print(fname)
    with open(fname, 'rb') as fh:
        feat_set, loc_set, img_set = pickle.load(fh)
        loc_set = loc_set.astype('int')
    
    feat_set = feat_set.T
    print('feat_set shape for category {}: {}'.format(cc, feat_set.shape))
    
    model = vMFMM(cluster_num,'k++')
    model.fit(feat_set, 30, max_it=100)
    
    save_path = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dict/dictionary_PASCAL3D+_{}_VGG16_pool2_K{}_vMFMM30.pickle'.format(cc, cluster_num)
    
    with open(save_path, 'wb') as fh:
        pickle.dump([model.p, model.mu, model.pi], fh)
    
#     ############## save examples ###################

#     num = 50
#     print('save top {} images for each cluster of {}'.format(num, cc))
#     example = [None for vc_i in range(cluster_num)]
#     for vc_i in range(cluster_num):
#         patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
#         sort_idx = np.argsort(-model.p[:,vc_i])[0:num]
#         for idx in range(num):
#             iloc = loc_set[:,sort_idx[idx]]
#             patch = img_set[iloc[0]][iloc[1]:iloc[3], iloc[2]:iloc[4], :]
#             patch_set[:,idx] = patch.flatten()

#         example[vc_i] = np.copy(patch_set)
#         if vc_i%10 == 0:
#             print(vc_i)

#     save_path2 = save_path.replace('.pickle','_example.pickle')
#     with open(save_path2, 'wb') as fh:
#         pickle.dump(example, fh)