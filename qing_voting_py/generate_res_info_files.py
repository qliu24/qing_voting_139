import math
import pickle
import scipy.io as sio
from scipy.spatial.distance import cdist
from FeatureExtractor import *
from config_voting import *

def generate_res_info_files(category_ls, set_type):
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
        
        # FCD_tmp = '/export/home/qliu24/qing_voting_data/intermediate/feat_alex'
        file_cache_feat = os.path.join(Feat['cache_dir'], '{0}_{1}_{2}_carVC_vMFMM30.pickle'.format(category, dataset_suffix, set_type))
        assert(os.path.isfile(file_cache_feat))
        with open(file_cache_feat, 'rb') as fh:
            _, r_set = pickle.load(fh)
            
        
        layer_feature_dist = [None for nn in range(img_num)]
        # layer_feature_dist_super = [None for nn in range(img_num)]
        sub_type = [None for nn in range(img_num)]
        view_point = [None for nn in range(img_num)]
        for nn in range(img_num):
            layer_feature_dist[nn] = r_set[nn]
            # layer_feature_dist_super[nn] = r_super_set[nn]
            file_anno = os.path.join(dir_anno, '{0}.mat'.format(img_list[nn][0]))
            assert(os.path.isfile(file_anno))
            mat_contents = sio.loadmat(file_anno)
            record = mat_contents['record']
            objects = record['objects']
            
            sub_type[nn] = objects[0,0]['subtype'][0,int(img_list[nn][1])-1][0]
            view_point[nn] = (objects[0,0]['viewpoint'][0,int(img_list[nn][1])-1]['azimuth_coarse'][0,0][0,0], \
                              objects[0,0]['viewpoint'][0,int(img_list[nn][1])-1]['elevation_coarse'][0,0][0,0])
            
            if nn%100 == 0:
                print(nn, end=' ')
                sys.stdout.flush()
            
            
        print('\n')
        
        sfile = VC['res_info'].format(category,set_type+'_carVC_vMFMM30')
        with open(sfile, 'wb') as fh:
            pickle.dump([layer_feature_dist, sub_type, view_point], fh)
            
if __name__=='__main__':
    # objs = ['car','aeroplane','bicycle','bus','motorbike','train']
    objs = ['car']
    generate_res_info_files(objs, 'train')