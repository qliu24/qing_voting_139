import glob
from config_voting import *
from comptScoreFuncs import *

def testVotingForBBoxes(model_category_ls, category_ls, set_type='test', model_type='mix'):
    for model_category in model_category_ls:
        Model_file = model_file_dic[model_category]
        
        assert(os.path.isfile(Model_file))
        with open(Model_file, 'rb') as fh:
            weights = pickle.load(fh)
            
        if model_type=='mix':
            weights, lbs_prior = weights
            cls_num = weights.shape[0]
            weight_objs = [weights[kk].reshape(model_dim_dic[model_category]).T for kk in range(cls_num)]
            logZs = [np.sum(np.log(np.exp(weights[kk])+1)) for kk in range(cls_num)]
            log_priors = [np.log(lbs_prior[kk]) for kk in range(cls_num)]
        elif model_type=='single':
            weight_obj = weights.T
            logZ = np.sum(np.log(np.exp(weights)+1))
        else:
            sys.exit('Error: unknown model type')
            
        for category in category_ls:
            file_list = Dataset['{0}_list'.format(set_type)].format(category)
            assert(os.path.isfile(file_list))
            
            with open(file_list, 'r') as fh:
                content = fh.readlines()
            
            img_list = [x.strip().split() for x in content]
            img_num = len(img_list)
            det = [dict() for nn in range(img_num)]
            nn = 0
            
            props_list = glob.glob(os.path.join(Feat['cache_dir'], \
                                                'props_feat_{0}_{1}_{2}_*.pickle'.format(category, dataset_suffix, set_type)))
            num_batch = len(props_list)
            
            print('compute voting scores...model:{0}, input:{1}'.format(model_category, category))
            for ii in range(num_batch):
                print('    for batch {0} of {1}:'.format(ii+1, num_batch), end=' ')
                sys.stdout.flush()
                
                props_file = os.path.join(Feat['cache_dir'], \
                                          'props_feat_{0}_{1}_{2}_{3}.pickle'.format(category, dataset_suffix, set_type, ii))
                
                assert(os.path.isfile(props_file))
                with open(props_file,'rb') as fh:
                    feat = pickle.load(fh)
                    
                for cnt_img in range(len(feat)):
                    det[nn]['img_path'] = feat[cnt_img]['img_path']
                    det[nn]['img_siz'] = feat[cnt_img]['img_siz']
                    det[nn]['box'] = feat[cnt_img]['box']
                    num_box = feat[cnt_img]['box'].shape[0]
                    det[nn]['score'] = np.zeros(num_box)
                    
                    for jj in range(num_box):
                        if model_type=='single':
                            det[nn]['score'][jj] = comptScores(feat[cnt_img]['r_super'][jj], weight_obj, logZ)
                        elif model_type=='mix':
                            det[nn]['score'][jj] = comptScoresM(feat[cnt_img]['r_super'][jj], weight_objs, logZs, log_priors)
                        else:
                            sys.exit('Error: unknown model type')
                            
                    if nn%10 == 0:
                        print(nn, end=' ')
                        sys.stdout.flush()
                    
                    nn += 1
                    
                print('\n', end='')
                    
            assert(nn == img_num)
            
            file_det_result = os.path.join(dir_det_result, 'props_det_{0}_{1}.pickle'.format(model_category, category))
            with open(file_det_result, 'wb') as fh:
                pickle.dump(det, fh)
                    
            
        
            