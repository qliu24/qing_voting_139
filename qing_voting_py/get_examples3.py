import numpy as np
import pickle

all_categories = ['car','aeroplane','bicycle','bus','motorbike','train']
category = 'train'

cluster_file = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dict/dictionary_PASCAL3D+_{}_VGG16_pool4_K150_vMFMM30.pickle'.format(category)

print('loading data...')

file_idx = all_categories.index(category)
Arf = 100
loc_set = np.zeros((5, 0), dtype='int')
img_set = []
fname = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/pool4_all_dumped_data{}.pickle'.format(file_idx)
with open(fname, 'rb') as fh:
    _, loc_set, img_set = pickle.load(fh)
    loc_set = loc_set.astype(int)
        

print('all loc_set')
print(loc_set.shape)
print('all images')
print(len(img_set))

with open(cluster_file, 'rb') as fh:
    model_p, model_mu, _ = pickle.load(fh)
    
K = model_mu.shape[0]

# the num of images for each cluster
num = 50
print('save top {0} images for each cluster'.format(num))
example = [None for nn in range(K)]
for k in range(K):
    patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
    sort_idx = np.argsort(-model_p[:,k])[0:num]
    for idx in range(num):
        iloc = loc_set[:,sort_idx[idx]]
        img_idx = int(iloc[0] - file_idx*1000)
        patch = img_set[img_idx][iloc[1]:iloc[3], iloc[2]:iloc[4], :]
        patch_set[:,idx] = patch.flatten()
        
    example[k] = np.copy(patch_set)
    if k%10 == 0:
        print(k)
        
        
save_path = cluster_file.replace('.pickle','_example.pickle')
with open(save_path, 'wb') as fh:
    pickle.dump(example, fh)

