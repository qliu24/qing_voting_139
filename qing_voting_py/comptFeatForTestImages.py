from config_voting import *
from scipy.spatial.distance import cdist
from FeatureExtractor import *
import scipy.io as sio

objs = ['car']
set_type = 'test'

assert(os.path.isfile(Dictionary_car))
with open(Dictionary_car, 'rb') as fh:
    _,centers = pickle.load(fh)

extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)
for category in objs:
    file_gt_obj_feat = os.path.join(dir_perf_eval, 'gt_feat_{0}_{1}_{2}.pickle'.format(category, dataset_suffix, set_type))
    
    n_pos = 0
    dir_img = Dataset['img_dir'].format(category)
    dir_anno = Dataset['anno_dir'].format(category)
    
    file_list = Dataset['{0}_list'.format(set_type)].format(category)
    assert(os.path.isfile(file_list))
    
    with open(file_list, 'r') as fh:
        content = fh.readlines()
        
    img_list = [x.strip().split() for x in content]
    img_num = len(img_list)
    img_num=100
    print('total number of images for {1}: {0}'.format(img_num, category))
    gt = [dict() for nn in range(img_num)]
    
    for nn in range(img_num):
        img_name = '{0}.JPEG'.format(img_list[nn][0])
        file_img = os.path.join(dir_img, img_name)
        gt[nn]['img_path'] = file_img
        
        img = cv2.imread(file_img)
        if file_img=='/export/home/qliu24/dataset/PASCAL3D+_release1.1/Images/motorbike_imagenet/n03790512_7145.JPEG':
            img = img.transpose(1,0,2)[:,::-1,:]
            
        hgt_img, wid_img, _ = img.shape
        
        file_anno = os.path.join(dir_anno, '{0}.mat'.format(img_list[nn][0]))
        assert(os.path.isfile(file_anno))
        
        mat_contents = sio.loadmat(file_anno)
        record = mat_contents['record']
        assert(record['imgsize'][0,0][0,0] == wid_img)
        assert(record['imgsize'][0,0][0,1] == hgt_img)
        objects = record['objects']
        
        for jj in range(objects[0,0]['class'].shape[1]):
            if objects[0,0]['class'][0,jj][0] == category:
                gt_bbox_cls = objects[0,0]['bbox'][0,jj][0]
                gt_bbox_cls = [max(math.ceil(gt_bbox_cls[0]), 1), max(math.ceil(gt_bbox_cls[1]), 1), \
                               min(math.floor(gt_bbox_cls[2]), wid_img), min(math.floor(gt_bbox_cls[3]), hgt_img)]
                
                gt[nn]['bbox'] = np.column_stack([gt[nn].get('bbox', np.zeros((4,0))), gt_bbox_cls])
                gt[nn]['diff'] = np.append(gt[nn].get('diff',[]), objects[0,0]['difficult'][0,jj][0,0])
                
                min_edge = np.min([gt_bbox_cls[2]-gt_bbox_cls[0], gt_bbox_cls[3]-gt_bbox_cls[1]])
                ratio = 224/min_edge
                img_resized = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
                layer_feature = extractor.extract_feature_image(img_resized)[0]
                gt[nn]['feat'] = gt[nn].get('feat',[]).append(layer_feature)
                
                iheight, iwidth = layer_feature.shape[0:2]
                assert(featDim == layer_feature.shape[2])
                layer_feature = layer_feature.reshape(-1, featDim)
                feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
                layer_feature = layer_feature/feat_norm
                
                dist = cdist(layer_feature, centers).reshape(iheight,iwidth,-1)
                gt[nn]['r'] = gt[nn].get('r',[]).append(dist)
                
        gt[nn]['det'] = np.zeros(gt[nn]['bbox'].shape[1]).astype(bool)
        
        if gt[nn]['bbox'].size==0:
            sys.exit('Empty gt bbox')
            
        n_pos += np.sum(1-gt[nn]['diff'])
        if nn % 10 == 0:
            print(nn, end=' ')
            sys.stdout.flush()
            
    print('\n', end='')
    with open(file_gt_obj_feat, 'wb') as fh:
        pickle.dump([gt, n_pos], fh)