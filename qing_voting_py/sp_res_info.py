from FeatureExtractor import *
import scipy.io as sio
from config_voting import *

def sp_res_info(category, occlusion = 0, use_bbox = 1):
    dir_anno = Dataset['anno_dir'].format(category)
    if occlusion==0:
        file_list = SP['img_list'].format(category)
        dir_img = Dataset['img_dir'].format(category)
    else:
        sys.exit('Have not implemented for occlusion other than 0')
    
    sp_dir = SP['anno_dir'].format(category)
    extractor = FeatureExtractor(cache_folder=model_cache_folder, which_layer=VC['layer'], which_snapshot=0)
    
    assert(os.path.isfile(file_list))
    with open(file_list, 'r') as fh:
        content = fh.readlines()
    
    img_list = [x.strip().split() for x in content]
    img_num = len(img_list)
    print('total number of images for {1}: {0}'.format(img_num, category))
    res_info = [dict() for ii in range(img_num)]
    
    for ii in range(img_num):
        if occlusion == 0:
            imgPath = os.path.join(dir_img, '{0}.JPEG'.format(img_list[ii][0]))
            annoPath = os.path.join(dir_anno, '{0}.mat'.format(img_list[ii][0]))
            spPath = os.path.join(sp_dir, '{0}.mat'.format(img_list[ii][0]))
            
            assert(os.path.isfile(spPath))
            mat_contents = sio.loadmat(spPath)
            spanno = mat_contents['anno'][int(img_list[ii][1])-1, 1]
            
            assert(os.path.isfile(imgPath))
            img = cv2.imread(imgPath)
            height, width = img.shape[0:2]
            
            assert(os.path.isfile(annoPath))
            mat_contents = sio.loadmat(annoPath)
            record = mat_contents['record']
            objects = record['objects']
            bbox = objects[0,0]['bbox'][0,int(img_list[ii][1])-1][0]
            bbox = [max(math.ceil(bbox[0]), 1), max(math.ceil(bbox[1]), 1), \
                    min(math.floor(bbox[2]), width), min(math.floor(bbox[3]), height)]
        else:
            sys.exit('Have not implemented for occlusion other than 0')
            
        if use_bbox==1:
            patch = img[bbox[1]-1: bbox[3], bbox[0]-1: bbox[2], :]
            scalePatch = myresize(patch, 224, 'short')
        else:
            sys.exit('Have not implemented for use_bbox other than 1')
            
        layer_feature = extractor.extract_feature_image(scalePatch)[0]
        
        res_info[ii]['img'] = scalePatch
        res_info[ii]['spanno'] = spanno
        res_info[ii]['res'] = layer_feature
        
        if ii%20==0:
            print(ii, end=' ', flush=True)
            
    print(' ')
    sp_res_info_file = os.path.join(SP['feat_dir'], '{0}_res_info_test.pickle'.format(category))
    with open(sp_res_info_file, 'wb') as fh:
        pickle.dump(res_info, fh)