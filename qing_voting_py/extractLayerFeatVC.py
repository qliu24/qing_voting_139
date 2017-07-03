import numpy as np
import math
import pickle
from scipy.spatial.distance import cdist
import scipy.io as sio
from FeatureExtractor import *
from config_voting import *

def extractLayerFeatVC(category_ls, set_type, scale_size=224):
    extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='alexnet', which_layer=VC['layer'], which_snapshot=93000)
    
    assert(os.path.isfile(Dictionary))
    with open(Dictionary, 'rb') as fh:
        _,centers = pickle.load(fh)
    
    for category in category_ls:
        dir_img = Dataset['img_dir'].format(category)
        dir_anno = Dataset['anno_dir'].format(category)
        
        file_list = Dataset['{0}_list'.format(set_type)].format(category)
        assert(os.path.isfile(file_list))
        
        with open(file_list, 'r') as fh:
            content = fh.readlines()
        
        img_list = [x.strip().split() for x in content]
        img_num = len(img_list)
        print('total number of images for {1}: {0}'.format(img_num, category))
        
        feat_set = [None for nn in range(img_num)]
        r_set = [None for nn in range(img_num)]
        for nn in range(img_num):
            file_img = os.path.join(dir_img, '{0}.JPEG'.format(img_list[nn][0]))
            assert(os.path.isfile(file_img))
            img = cv2.imread(file_img)
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show()
            
            height, width = img.shape[0:2]
            
            file_anno = os.path.join(dir_anno, '{0}.mat'.format(img_list[nn][0]))
            assert(os.path.isfile(file_anno))
            mat_contents = sio.loadmat(file_anno)
            record = mat_contents['record']
            objects = record['objects']
            bbox = objects[0,0]['bbox'][0,int(img_list[nn][1])-1][0]
            bbox = [max(math.ceil(bbox[0]), 1), max(math.ceil(bbox[1]), 1), \
                    min(math.floor(bbox[2]), width), min(math.floor(bbox[3]), height)]
            patch = img[bbox[1]-1: bbox[3], bbox[0]-1: bbox[2], :]
            # patch = cv2.resize(patch, (scale_size, scale_size))
            patch = myresize(patch, scale_size, 'short')
            
            layer_feature = extractor.extract_feature_image(patch, is_gray=True)[0]
            iheight, iwidth = layer_feature.shape[0:2]
            assert(featDim == layer_feature.shape[2])
            feat_set[nn] = layer_feature
            
            layer_feature = layer_feature.reshape(-1, featDim)
            feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
            layer_feature = layer_feature/feat_norm
            
            dist = cdist(layer_feature, centers).reshape(iheight,iwidth,-1)
            assert(dist.shape[2]==centers.shape[0]);
            r_set[nn] = dist
            
            if nn%100 == 0:
                print(nn, end=' ')
                sys.stdout.flush()
            
            
        print('\n')
        
        file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}_gray200.pickle'.format(category, dataset_suffix, set_type))
        with open(file_cache_feat, 'wb') as fh:
            pickle.dump([feat_set, r_set], fh)
            
            
if __name__=='__main__':
    objs = ['car','aeroplane','bicycle','bus','motorbike','train']
    # objs = ['car']
    extractLayerFeatVC(objs, 'train')