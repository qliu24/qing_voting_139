from scipy.spatial.distance import cdist
from copy import *
from FeatureExtractor import *
from config_voting import *

scale_size = 224

assert(os.path.isfile(Dict['file_list']))
with open(Dict['file_list'], 'r') as fh:
    image_path = [ff.strip() for ff in fh.readlines()]
    
# img_num = len(image_path)
img_num=1000
print('total images number : {0}'.format(img_num))

assert(os.path.isfile(Dictionary_car))
with open(Dictionary_car, 'rb') as fh:
    _,centers = pickle.load(fh)

extractor = FeatureExtractor(cache_folder=model_cache_folder, which_layer=VC['layer'], which_snapshot=0)

feat_set = [None for nn in range(img_num)]
img_set = [None for nn in range(img_num)]
r_set = [None for nn in range(img_num)]
for nn in range(img_num):
    img = cv2.imread(image_path[nn])
    # img = cv2.resize(img, (scale_size, scale_size))
    img = myresize(img, scale_size, 'short')
    img_set[nn] = deepcopy(img)
    layer_feature = extractor.extract_feature_image(img)[0]
    feat_set[nn] = deepcopy(layer_feature)
    
    height, width = layer_feature.shape[0:2]
    assert(featDim == layer_feature.shape[2])
    
    layer_feature = layer_feature.reshape(-1, featDim)
    feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
    layer_feature = layer_feature/feat_norm
    
    dist = cdist(layer_feature, centers).reshape(height,width,-1)
    assert(dist.shape[2]==centers.shape[0]);
    r_set[nn] = dist
    
    
    if nn%50 == 0:
        print(nn, end=' ')
        sys.stdout.flush()
        
print('\n')
        
file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}.pickle'.format('car', dataset_suffix, 'both'))
with open(file_cache_feat, 'wb') as fh:
    pickle.dump([feat_set,img_set,r_set], fh)


