from config_voting import *
def merge_det(list1, list2):
    for model_category in list1:
        save_file = os.path.join(dir_det_result, 'props_det_{0}_{1}.pickle'.format(model_category, 'all'))
        det = []
        
        for category in list2:
            file_det_result = os.path.join(dir_det_result, 'props_det_{0}_{1}.pickle'.format(model_category, category))
            assert(os.path.isfile(file_det_result))
            with open(file_det_result, 'rb') as fh:
                det_i = pickle.load(fh)
                
            for nn in range(len(det_i)):
                det_i[nn]['cat'] = category
            
            det += det_i
            
        with open(save_file, 'wb') as fh:
            pickle.dump(det, fh)