import numpy as np
import math
import pickle
import scipy.io as sio
import sys
from FeatureExtractor import *
from config_voting import *

def comptFeatForBBoxes(category_ls, set_type):
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
        
        file_nm = 'bbox_props_{0}_{1}_{2}_2.mat'.format(category, dataset_suffix, set_type)
        file_bbox_proposals = os.path.join(Data['root_dir'], file_nm)
        assert(os.path.isfile(file_bbox_proposals))
        mat_contents = sio.loadmat(file_bbox_proposals)
        Box = mat_contents['Box']
        # Box = [mat_contents['Box'][0,nn][0] for nn in range(mat_contents['Box'].shape[1])]
        if category == 'car':
            assert(Box.shape[1]==img_num)
        else:
            assert(Box.shape[0]==img_num)
        
        num_batch = math.ceil(img_num/Feat['num_batch_img'])
        
        for ii in range(num_batch):
            file_nm='props_feat_{0}_{1}_{2}_{3}.pickle'.format(category, dataset_suffix, set_type, ii)
            file_cache_feat_batch = os.path.join(Feat['cache_dir'], file_nm)
            img_start_id = Feat['num_batch_img']*ii
            img_end_id = min(Feat['num_batch_img']*(ii+1), img_num)
            
            print('stack {0} ({1} ~ {2}):'.format(ii, img_start_id, img_end_id))
            
            feat = [dict() for nn in range(img_end_id-img_start_id)]
            cnt_img = -1
            for nn in range(img_start_id, img_end_id):
                cnt_img += 1
                file_img = os.path.join(dir_img, '{0}.JPEG'.format(img_list[nn][0]))
                assert(os.path.isfile(file_img))
                img = cv2.imread(file_img)
                height, width, _ = img.shape
                
                if category=='car':
                    assert(Box['anno'][0,nn]['height'][0,0][0,0] == height)
                    assert(Box['anno'][0,nn]['width'][0,0][0,0] == width)
                    boxes = Box['boxes'][0,nn]
                else:
                    assert(Box[nn,0]['name'][0,0][0]==img_list[nn][0])
                    boxes = Box[nn,0]['boxes'][0,0]
                
                boxes = boxes[0:min(Feat['max_num_props_per_img'], boxes.shape[0]), :]
                num_box = boxes.shape[0]
                
                feat[cnt_img]['img_path'] = file_img
                feat[cnt_img]['img_siz'] = [height, width]
                feat[cnt_img]['box'] = boxes[:,0:4]
                feat[cnt_img]['feat'] = [None for jj in range(num_box)]
                
                for jj in range(num_box):
                    bbox = boxes[jj, 0:4]
                    bbox = [max(math.ceil(bbox[0]), 1), max(math.ceil(bbox[1]), 1), \
                            min(math.floor(bbox[2]), width), min(math.floor(bbox[3]), height)]
                    
                    patch = img[bbox[1]-1: bbox[3], bbox[0]-1: bbox[2], :]
                    feat[cnt_img]['feat'][jj] = extractor.extract_feature_image(patch)[0]
                
                if cnt_img%10 == 0:
                    print(cnt_img, end=' ')
                    sys.stdout.flush()
            
            print('\n')
            with open(file_cache_feat_batch, 'wb') as fh:
                pickle.dump(feat, fh)


