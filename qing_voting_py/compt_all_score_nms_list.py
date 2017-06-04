from config_voting import *
from nms import nms

category = 'all'
objs = all_categories+all_bgs
savefile = os.path.join(dir_det_result,'all_score_nms_list.pickle')

tmp = []
print('loading files')
for model_category in objs:
    filename = os.path.join(dir_det_result, 'props_det_{0}_{1}.pickle'.format(model_category, category))
    with open(filename, 'rb') as fh:
        det = pickle.load(fh)
        
    tmp.append([det[nn]['score'] for nn in range(len(det))])
    
img_num = len(tmp[0])
print('total images number: {0}'.format(img_num))
score_all = [np.zeros((len(tmp[0][nn]),0)) for nn in range(img_num)]
for nn in range(img_num):
    for tt in range(len(tmp)):
        score_all[nn] = np.column_stack([score_all[nn], tmp[tt][nn]])
        
score_rst = [np.zeros((len(tmp[0][nn]),0)) for nn in range(img_num)]
score_rst2 = [np.zeros((len(tmp[0][nn]),0)) for nn in range(img_num)]
for nn in range(img_num):
    for oi in range(6):
        score_rst[nn] = np.column_stack([score_rst[nn], \
                                         score_all[nn][:,oi] - np.max(score_all[nn][:,np.arange(len(objs))!=oi], axis=1)])
        
        score_rst2[nn] = np.column_stack([score_rst2[nn], \
                                          score_all[nn][:,oi] - np.max(score_all[nn][:,6:], axis=1)])
        
num_list_all = [None for nn in range(img_num)]
num_list_all2 = [None for nn in range(img_num)]
for nn in range(img_num):
    boxes = det[nn]['box']
    assert(score_rst[nn].shape[0] == boxes.shape[0])
    score_highest = np.max(score_rst[nn],axis=1)
    # adhoc thing
    '''
    si = np.argsort(-score_highest)
    height, width = det[nn]['img_siz']
    topn = 5
    bbox_area = np.zeros(topn)
    for mm in range(topn):
        bbmm = boxes[si[mm]]
        bbmm = [max(math.ceil(bbmm[0]), 1), max(math.ceil(bbmm[1]), 1), \
                min(math.floor(bbmm[2]), width), min(math.floor(bbmm[3]), height)]
        bbox_area[mm] = (bbmm[2]-bbmm[0])*(bbmm[3]-bbmm[1])
        
    biggest_i = np.argmax(bbox_area)
    score_highest[si[biggest_i]] += 100
    '''
    
    score_highest2 = np.max(score_rst2[nn],axis=1)
    # adhoc thing
    '''
    si = np.argsort(-score_highest2)
    bbox_area = np.zeros(topn)
    for mm in range(topn):
        bbmm = boxes[si[mm]]
        bbmm = [max(math.ceil(bbmm[0]), 1), max(math.ceil(bbmm[1]), 1), \
                min(math.floor(bbmm[2]), width), min(math.floor(bbmm[3]), height)]
        bbox_area[mm] = (bbmm[2]-bbmm[0])*(bbmm[3]-bbmm[1])
        
    biggest_i = np.argmax(bbox_area)
    score_highest2[si[biggest_i]] += 100
    '''
    num_list_all[nn] = nms(np.column_stack([boxes, score_highest]), nms_thrh)
    num_list_all2[nn] = nms(np.column_stack([boxes, score_highest2]), nms_thrh)
    

for dd in det:
    del dd['score']
    
print('saving results to {0}'.format(savefile))
with open(savefile,'wb') as fh:
    pickle.dump([score_rst, score_rst2, num_list_all, num_list_all2, det], fh)

