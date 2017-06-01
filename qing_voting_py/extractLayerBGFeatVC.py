import numpy as np
import math
import pickle
from scipy.spatial.distance import cdist
import scipy.io as sio
from FeatureExtractor import *
from config_voting import *

def extractLayerBGFeatVC(category_ls, set_type, scale_size=224, bg_per_img=3):
    extractor = FeatureExtractor(cache_folder=model_cache_folder, which_layer=VC['layer'], which_snapshot=0)
    
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
        
        no_bg_num=0
        
        feat_set = []
        r_set = []
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
            total_bboxs = objects[0,0]['bbox'].shape[1]
            
            bbox = [max(math.ceil(bbox[0]), 1), max(math.ceil(bbox[1]), 1), \
                    min(math.floor(bbox[2]), width), min(math.floor(bbox[3]), height)]
            bbox_h = bbox[3] - bbox[1] + 1
            bbox_w = bbox[2] - bbox[0] + 1
            
            for mm in range(bg_per_img):
                find_bg = np.zeros(total_bboxs)
                rdm_cnt = 0
                while not np.all(find_bg) and rdm_cnt < 10000:
                    rdm_cnt += 1
                    find_bg = np.zeros(total_bboxs)
                    r_h = np.random.randint(height - bbox_h + 1)
                    r_w = np.random.randint(width - bbox_w + 1)
                    bbox_bg = [r_w, r_h, r_w+bbox_w-1, r_h+bbox_h-1]
                    
                    for bbi in range(total_bboxs):
                        bbox = objects[0,0]['bbox'][0,bbi][0]
                        bbox = [max(math.ceil(bbox[0]), 1), max(math.ceil(bbox[1]), 1), \
                                min(math.floor(bbox[2]), width), min(math.floor(bbox[3]), height)]
                        bbox_area = (bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1)
                        over_1 = max(bbox[0], bbox_bg[0])
                        over_3 = min(bbox[2], bbox_bg[2])
                        over_2 = max(bbox[1], bbox_bg[1])
                        over_4 = min(bbox[3], bbox_bg[3])
                        if over_1 >= over_3 or over_2 >= over_4:
                            find_bg[bbi] = True
                        else:
                            over_area = (over_3-over_1+1)*(over_4-over_2+1)
                            if over_area/bbox_area < 0.6:
                                find_bg[bbi] = True
                                
                if rdm_cnt == 10000:
                    no_bg_num += 1
                    continue
                    
                patch = img[bbox_bg[1]: bbox_bg[3]+1, bbox_bg[0]: bbox_bg[2]+1, :]
                # patch = cv2.resize(patch, (scale_size, scale_size))
                patch = myresize(patch, scale_size, 'short')
                
                layer_feature = extractor.extract_feature_image(patch)[0]
                iheight, iwidth = layer_feature.shape[0:2]
                assert(featDim == layer_feature.shape[2])
                feat_set.append(layer_feature)
                
                layer_feature = layer_feature.reshape(-1, featDim)
                feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
                layer_feature = layer_feature/feat_norm
                
                dist = cdist(layer_feature, centers).reshape(iheight,iwidth,-1)
                assert(dist.shape[2]==centers.shape[0])
                r_set.append(dist)
                
            if nn%100 == 0:
                print(nn, end=' ')
                sys.stdout.flush()
                
        print('\n')
        print('total number of bg for {1}: {0}'.format(len(feat_set), category))
        
        file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}_bg.pickle'.format(category, dataset_suffix, set_type))
        with open(file_cache_feat, 'wb') as fh:
            pickle.dump([feat_set, r_set], fh)


