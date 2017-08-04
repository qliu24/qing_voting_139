from config_voting_ILSVRC12 import *
import tensorflow as tf
from tensorflow.python.client import timeline
from datetime import datetime
from copy import *
from FeatureExtractor import FeatureExtractor

check_num = 2000  # save how many images to one file
samp_size = 50  # number of features per image
scale_size = 224

# Specify the dataset
assert(os.path.isfile(Dict['file_list']))
with open(Dict['file_list'], 'r') as fh:
    image_path = [ff.strip() for ff in fh.readlines()]
    
img_num = len(image_path)
print('total images number : {0}'.format(img_num))

extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)

res = np.zeros((featDim, 0))
loc_set = np.zeros((5, 0))

for ii in range(img_num):
    img = cv2.imread(os.path.join(Dict['file_dir'], image_path[ii]))
    # img = cv2.resize(img, (scale_size, scale_size))
    img = myresize(img, scale_size, 'short')
    
    tmp = extractor.extract_feature_image(img)[0]
    assert(tmp.shape[2]==featDim)
    height, width = tmp.shape[0:2]
    tmp = tmp[offset:height - offset, offset:width - offset, :]
    ntmp = np.transpose(tmp, (2, 0, 1))
    gtmp = ntmp.reshape(ntmp.shape[0], -1)
    if gtmp.shape[1] >= samp_size:
        rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size]
    else:
        rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size-gtmp.shape[1]]
        rand_idx = np.append(range(gtmp.shape[1]), rand_idx)
        
    res = np.column_stack((res, deepcopy(gtmp[:, rand_idx])))
    for rr in rand_idx:
        ihi, iwi = np.unravel_index(rr, (height - 2 * offset, width - 2 * offset))
        hi = Astride * (ihi + offset) - Apad
        wi = Astride * (iwi + offset) - Apad
        assert (hi >= 0)
        assert (hi <= img.shape[0] - Arf)
        assert (wi >= 0)
        assert (wi <= img.shape[1] - Arf)
        loc_set = np.column_stack((loc_set, [ii, hi, wi, hi+Arf, wi+Arf]))
    
    if (ii + 1) % check_num == 0 or ii == img_num - 1:
        print('saving batch {0}/{1}'.format(ii//check_num+1, math.ceil(img_num/check_num)))
        fnm = Dict['cache_path']+str(ii//check_num)+'.pickle'
        with open(fnm, 'wb') as fh:
            pickle.dump([res, loc_set], fh)
        
        res = np.zeros((featDim, 0))
        loc_set = np.zeros((5, 0))
        
    if ii%50==0:
        print(ii, end=' ', flush=True)

