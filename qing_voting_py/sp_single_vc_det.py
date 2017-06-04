import scipy.io as sio
from scipy.spatial.distance import cdist
from nms import nms
from VOCap import VOCap
from config_voting import *

def sp_single_vc_det(category, bbox = 1):
    sp_single_vc_det_file = os.path.join(SP['result'], '{0}_sp_test_carvc.pickle'.format(category))
    
    sp_res_info_file = os.path.join(SP['feat_dir'], '{0}_res_info_test.pickle'.format(category))
    with open(sp_res_info_file, 'rb') as fh:
        res_info = pickle.load(fh)
        
    assert(os.path.isfile(Dictionary_car))
    with open(Dictionary_car, 'rb') as fh:
        _,centers = pickle.load(fh)
    
    '''
    assert(os.path.isfile(Dictionary_super))
    with open(Dictionary_super, 'rb') as fh:
        _,centers_super = pickle.load(fh)
    '''
    
    VCnum = centers.shape[0]
    
    img_num = len(res_info)
    # img_num = 20
    print('total number of images for {1}: {0}'.format(img_num, category))
    sp_detection = [np.zeros((0,6)) for jj in range(VCnum)]
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
        assert(dist.shape[0]==iheight*iwidth)
        '''
        dist = dist.reshape(iheight, iwidth, -1)
        super_f = np.zeros((iheight-patch_size[0]+1, iwidth-patch_size[1]+1, patch_size[0]*patch_size[1]*centers.shape[0]))
        for hh in range(0, iheight-patch_size[0]+1):
            for ww in range(0, iwidth-patch_size[1]+1):
                super_f[hh,ww,:] = dist[hh:hh+patch_size[0], ww:ww+patch_size[1], :].reshape(-1,)
        
        assert(super_f.shape[2] == centers_super.shape[1])
        
        super_f = super_f.reshape(-1, super_f.shape[2])
        super_f_norm = np.sqrt(np.sum(super_f**2, 1)).reshape(-1,1)
        super_f = super_f/super_f_norm
        
        # dist2 = cdist(super_f, centers_super).reshape(iheight-patch_size[0]+1,iwidth-patch_size[1]+1,-1)
        dist2 = cdist(super_f, centers_super)
        assert(dist2.shape[1]==centers_super.shape[0])
        
        r_list, c_list = np.unravel_index(list(range(dist2.shape[0])), (iheight-patch_size[0]+1,iwidth-patch_size[1]+1))
        r_list = Astride * (r_list+int(patch_size[0]/2)) + Arf/2 - Apad
        c_list = Astride * (c_list+int(patch_size[1]/2)) + Arf/2 - Apad
        bb_loc = np.column_stack([c_list-Arf/2, r_list-Arf/2, c_list+Arf/2, r_list+Arf/2])
        '''
        for jj in range(dist.shape[1]):
            bb_loc_ = np.column_stack([bb_loc, -dist[:,jj]])
            nms_list = nms(bb_loc_, 0.05)
            res_i = np.column_stack([nn * np.ones(len(nms_list)), bb_loc_[nms_list, :]])
            sp_detection[jj] = np.concatenate([sp_detection[jj], res_i], axis=0)
            
        
        if nn%20 == 0:
            print(nn, end=' ', flush=True)
            
    print(' ')
    
    SPnum = res_info[nn]['spanno'].shape[0]
    kp_pos = np.zeros(SPnum)
    for nn in range(img_num):
        for kk in range(SPnum):
            kp_pos[kk] += res_info[nn]['spanno'][kk,0].shape[0]
    
    ap = np.zeros((VCnum, SPnum))
    pr = [[[] for kk in range(SPnum)] for ii in range(VCnum)]
    for ii in range(VCnum):
        tot = sp_detection[ii].shape[0]
        sort_idx = np.argsort(-sp_detection[ii][:,5])
        id_list = sp_detection[ii][sort_idx, 0].astype(int)
        col_list = (sp_detection[ii][sort_idx, 1] + sp_detection[ii][sort_idx, 3])/2 + 1
        row_list = (sp_detection[ii][sort_idx, 2] + sp_detection[ii][sort_idx, 4])/2 + 1
        
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
                    
                    if np.sqrt((col_c-xx)**2 + (row_c-yy)**2) < min_dist:
                        min_dist = np.sqrt((col_c-xx)**2 + (row_c-yy)**2)
                        min_idx = jj
                
                if min_dist < Eval['dist_thresh'] and flag[img_id, min_idx] == 0:
                    tp[thresh] = 1
                    flag[img_id, min_idx] = 1 
                else:
                    fp[thresh] = 1
            
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / kp_pos[kk]
            prec = tp / (fp+tp)
            ap[ii, kk] = VOCap(rec[10:], prec[10:])
            pr[ii][kk] = np.array([rec, prec])
            
    mAP = np.mean(np.max(ap, axis=0))
    print('mAP: %4.1f'%(mAP*100))
    
    with open(sp_single_vc_det_file, 'wb') as fh:
        pickle.dump([ap, pr], fh)
        
        
    ap05 = np.zeros((VCnum, SPnum))
    pr05 = [[[] for kk in range(SPnum)] for ii in range(VCnum)]
    for ii in range(VCnum):
        tot = sp_detection[ii].shape[0]
        sort_idx = np.argsort(-sp_detection[ii][:,5])
        id_list = sp_detection[ii][sort_idx, 0].astype(int)
        col_list = (sp_detection[ii][sort_idx, 1] + sp_detection[ii][sort_idx, 3])/2 + 1
        row_list = (sp_detection[ii][sort_idx, 2] + sp_detection[ii][sort_idx, 4])/2 + 1
        
        for kk in range(SPnum):
            tp = np.zeros(tot)
            fp = np.zeros(tot)
            flag = np.zeros((img_num, 20))
            for thresh in range(tot):
                img_id = id_list[thresh]
                col_c = col_list[thresh]
                row_c = row_list[thresh]
                
                max_iou = 0
                inst = res_info[img_id]['spanno'][kk, 0]
                img_height, img_width = res_info[img_id]['img'].shape[0:2]
                for jj in range(inst.shape[0]):
                    if bbox==1:
                        xx = (inst[jj,0]+inst[jj,2])/2
                        yy = (inst[jj,1]+inst[jj,3])/2
                    else:
                        sys.exit('Have not implemented for bbox other than 1')
                    
                    x1=max(1,xx-50)
                    x2=min(img_width,xx+50)
                    y1=max(1,yy-50)
                    y2=min(img_height,yy+50)
                    c1=max(1,col_c-50)
                    c2=min(img_width,col_c+50)
                    r1=max(1,row_c-50)
                    r2=min(img_height,row_c+50)
                    intersection=max(min(x2,c2)-max(x1,c1),0)*max(min(y2,r2)-max(y1,r1),0)
                    iou=intersection/((x2-x1)*(y2-y1)+(c2-c1)*(r2-r1)-intersection)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = jj
                
                if max_iou >= Eval['iou_thresh'] and flag[img_id, max_idx] == 0:
                    tp[thresh] = 1
                    flag[img_id, max_idx] = 1 
                else:
                    fp[thresh] = 1
            
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / kp_pos[kk]
            prec = tp / (fp+tp)
            ap05[ii, kk] = VOCap(rec[10:], prec[10:])
            pr05[ii][kk] = np.array([rec, prec])
            
    mAP = np.mean(np.max(ap05, axis=0))
    print('mAP: %4.1f'%(mAP*100))
    
    with open(sp_single_vc_det_file, 'wb') as fh:
        pickle.dump([ap, pr, ap05, pr05], fh)
