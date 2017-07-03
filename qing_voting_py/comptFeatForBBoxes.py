import scipy.io as sio
from scipy.spatial.distance import cdist
from FeatureExtractor import *
from config_voting import *

def comptFeatForBBoxes(category_ls, set_type='test'):
    extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)
    
    assert(os.path.isfile(Dictionary_car))
    with open(Dictionary_car, 'rb') as fh:
        _,centers = pickle.load(fh)
        
    assert(centers.shape[0]==VC['num_car'])
    '''
    assert(os.path.isfile(Dictionary_super))
    with open(Dictionary_super, 'rb') as fh:
        _,centers_super = pickle.load(fh)
        
    assert(centers_super.shape[0]==VC['num_super'])
    '''
    
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
                # weird rotated image
                if file_img=='/export/home/qliu24/dataset/PASCAL3D+_release1.1/Images/motorbike_imagenet/n03790512_7145.JPEG':
                    img = img.transpose(1,0,2)[:,::-1,:]
                    
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
                feat[cnt_img]['r'] = [None for jj in range(num_box)]
                # feat[cnt_img]['r_super'] = [None for jj in range(num_box)]
                
                for jj in range(num_box):
                    bbox = boxes[jj, 0:4]
                    bbox = [max(math.ceil(bbox[0]), 1), max(math.ceil(bbox[1]), 1), \
                            min(math.floor(bbox[2]), width), min(math.floor(bbox[3]), height)]
                    
                    patch = img[bbox[1]-1: bbox[3], bbox[0]-1: bbox[2], :]
                    patch = myresize(patch, scale_size, 'short')
                    layer_feature = extractor.extract_feature_image(patch)[0]
                    feat[cnt_img]['feat'][jj] = layer_feature
                    
                    
                    iheight, iwidth = layer_feature.shape[0:2]
                    assert(featDim == layer_feature.shape[2])
                    layer_feature = layer_feature.reshape(-1, featDim)
                    feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
                    layer_feature = layer_feature/feat_norm
                    
                    dist = cdist(layer_feature, centers).reshape(iheight,iwidth,-1)
                    assert(dist.shape[2]==centers.shape[0])
                    
                    feat[cnt_img]['r'][jj] = dist
                    
                    '''
                    super_f = np.zeros((iheight-patch_size[0]+1, iwidth-patch_size[1]+1, patch_size[0]*patch_size[1]*VC['num']))
                    for hh in range(0, iheight-patch_size[0]+1):
                        for ww in range(0, iwidth-patch_size[1]+1):
                            super_f[hh,ww,:] = dist[hh:hh+patch_size[0], ww:ww+patch_size[1], :].reshape(-1,)
                            
                    super_f = super_f.reshape(-1, super_f.shape[2])
                    super_f_norm = np.sqrt(np.sum(super_f**2, 1)).reshape(-1,1)
                    super_f = super_f/super_f_norm
                    
                    dist2 = cdist(super_f, centers_super).reshape(iheight-patch_size[0]+1,iwidth-patch_size[1]+1,-1)
                    assert(dist2.shape[2]==centers_super.shape[0])
                    feat[cnt_img]['r_super'][jj] = dist2
                    '''
                
                if cnt_img%10 == 0:
                    print(cnt_img, end=' ')
                    sys.stdout.flush()
            
            
            print('\n', end='')
            with open(file_cache_feat_batch, 'wb') as fh:
                pickle.dump(feat, fh)
                

if __name__=="__main__":
    objs = ['motorbike','train']
    comptFeatForBBoxes(objs, 'test')
    


