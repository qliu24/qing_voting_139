import scipy.io as sio
from scipy.spatial.distance import cdist
from nms import nms
from config_voting import *

category = 'car'
bbox = 1
sp_res_info_file = os.path.join(SP['feat_dir'], '{0}_res_info_test.pickle'.format(category))
with open(sp_res_info_file, 'rb') as fh:
    res_info = pickle.load(fh)
    
assert(os.path.isfile(Dictionary))
with open(Dictionary, 'rb') as fh:
    _,centers = pickle.load(fh)
    
assert(os.path.isfile(Dictionary_super))
with open(Dictionary_super, 'rb') as fh:
    _,centers_super = pickle.load(fh)
    
img_num = len(res_info)
sp_detection = [np.zeros((0,6)) for jj in range(VC['num'])]

for nn in range(img_num):
    layer_feature = res_info[nn]['res']
    iheight, iwidth = layer_feature.shape[0:2]
    assert(featDim == layer_feature.shape[2])
    
    r_list, c_list = np.unravel_index(list(range(iheight*iwidth)), (iheight, iwidth))
    r_list = Astride * r_list + Arf/2 - Apad
    c_list = Astride * c_list + Arf/2 - Apad
    bb_loc = np.column_stack([c_list-Arf/2, r_list-Arf/2, c_list+Arf/2, r_list+Arf/2])
    
    layer_feature = layer_feature.reshape(-1, featDim)
    feat_norm = np.sqrt(np.sum(layer_feature**2, 1)).reshape(-1,1)
    layer_feature = layer_feature/feat_norm
    
    dist = cdist(layer_feature, centers)
    assert(dist.shape[1]==centers.shape[0])
    
    for jj in range(dist.shape[1]):
        bb_loc_ = np.column_stack([bb_loc, -dist[:,jj]])
        nms_list = nms(bb_loc_, 0.05)
        res_i = np.column_stack([nn * np.ones(len(nms_list)), bb_loc_[nms_list, :]])
        sp_detection[jj] = np.concatenate([sp_detection[jj], res_i], axis=0)
    
    '''
    super_f = np.zeros((iheight-patch_size[0]+1, iwidth-patch_size[1]+1, patch_size[0]*patch_size[1]*VC['num']))
    for hh in range(0, iheight-patch_size[0]+1):
        for ww in range(0, iwidth-patch_size[1]+1):
            super_f[hh,ww,:] = dist[hh:hh+patch_size[0], ww:ww+patch_size[1], :].reshape(-1,)
    
    assert(super_f.shape[2] == centers_super.shape[1])
    
    super_f = super_f.reshape(-1, super_f.shape[2])
    super_f_norm = np.sqrt(np.sum(super_f**2, 1)).reshape(-1,1)
    super_f = super_f/super_f_norm
    
    dist2 = cdist(super_f, centers_super).reshape(iheight-patch_size[0]+1,iwidth-patch_size[1]+1,-1)
    assert(dist2.shape[2]==centers_super.shape[0])
    '''
    if nn%20 == 0:
        print(nn, end=' ', flush=True)
        

print(' ')

SPnum = res_info[nn]['spanno'].shape[0]
kp_pos = np.zeros(SPnum)
for nn in range(img_num):
    for kk in range(SPnum):
        kp_pos[kk] += res_info[nn]['spanno'][kk,0].shape[0]
        
ap = np.zeros((VC['num'], SPnum))
pr = np.zeros((VC['num'], SPnum, 2))
Eval.dist_thresh = 56
for ii in range(VC['num']):
    print('VC number {0}'.format(ii))
    tot = sp_detection[ii].shape[0]
    sort_idx = np.argsort(-sp_detection[ii][:,5], )
    id_list = sp_detection[ii][sort_dix, 0]
    col_list = (sp_detection[ii][sort_dix, 1] + sp_detection[ii][sort_dix, 3])/2
    row_list = (sp_detection[ii][sort_dix, 2] + sp_detection[ii][sort_dix, 4])/2
    
    for kk in range(SPnum):
        tp = np.zeros(tot)
        fp = np.zeros(tot)
        flag = np.zeros((img_num, 20))
        # TODO: Here we use the arbitrary number 20, assuming no sp has been
        # labelled for over 20 times on a single image
        
        for thresh in range(tot):
            img_id = id_list[thresh]
            col_c = col_list[thresh]
            row_c = row_list[thresh]
            
            min_dist = np.inf
            inst = res_info[img_id]['spanno'][kk, 0]
            for jj in range(inst.shape[0]):
                if bbox==1:
                    xx = (inst[jj,0]+inst[jj,2])/2
                    yy = (inst[jj,1]+inst[jj,3])/2
                else:
                    sys.exit('Have not implemented for bbox other than 1')
                    
                if sqrt((col_c-xx)**2 + (row_c-yy)**2) < min_dist:
                    min_dist = sqrt((col_c-xx)**2 + (row_c-yy)**2)
                    min_idx = jj
                    
            if min_dist < Eval.dist_thresh and flag[img_id, min_idx] == 0
                tp[thresh] = 1
                flag[img_id, min_idx] = 1 
            else:
                fp[thresh] = 1
                
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / kp_pos[kk]
        prec = tp / (fp+tp)
        ap[ii, kk] = VOCap(rec[10:], prec[10:])
        pr[ii, kk] = np.array([rec, prec])
        
mAP = np.mean(np.max(ap, axis=0))
                