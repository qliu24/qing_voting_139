import numpy as np
import math
import pickle
import scipy.io as sio
from FeatureExtractor import *
from config_voting import *

def extractLayerFeat(category_ls, set_type):
    extractor = FeatureExtractor(cache_folder=model_cache_folder, which_layer=VC['layer'], which_snapshot=25000)
    
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
        for nn in range(img_num):
            file_img = os.path.join(dir_img, '{0}.JPEG'.format(img_list[nn][0]))
            assert(os.path.isfile(file_img))
            img = cv2.imread(file_img)
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show()
            
            height, width, _ = img.shape
            
            file_anno = os.path.join(dir_anno, '{0}.mat'.format(img_list[nn][0]))
            assert(os.path.isfile(file_anno))
            mat_contents = sio.loadmat(file_anno)
            record = mat_contents['record']
            objects = record['objects']
            bbox = objects[0,0]['bbox'][0,int(img_list[nn][1])-1][0]
            bbox = [max(math.ceil(bbox[0]), 1), max(math.ceil(bbox[1]), 1), \
                    min(math.floor(bbox[2]), width), min(math.floor(bbox[3]), height)]
            patch = img[bbox[1]-1: bbox[3], bbox[0]-1: bbox[2], :]
            
            feat_set[nn] = extractor.extract_feature_image(patch)[0]
            
            if nn%100 == 0:
                print(nn, end=' ')
            
            
        print('\n')
        
        if not os.path.exists(Feat['cache_dir']):
            os.makedirs(Feat['cache_dir'])
            
        file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}.pickle'.format(category, dataset_suffix, set_type))
        with open(file_cache_feat, 'wb') as fh:
            pickle.dump(feat_set, fh)

