import math
import numpy as np
from nms import nms

def get_score(lfd, lfb, heat_map, vc_templates):
    heat_map_min = 1/350
    hm_BG_val = np.log(heat_map_min)
    
    total_score = 0
    max_0, max_1, max_2 = heat_map.shape
    
    rll,cll = lfd.shape[0:2]
    diff_r = np.floor((max_0-rll)/2).astype(int)
    diff_c = np.floor((max_1-cll)/2).astype(int)
    if diff_r >= 0:
        hm_ll = heat_map[diff_r:diff_r+rll,:,:]
        lfd_adj = lfd
        lfb_adj = lfb
    else:
        hm_ll = heat_map
        diff_r = -diff_r
        lfd_adj = lfd[diff_r:diff_r+max_0,:,:]
        lfb_adj = lfb[diff_r:diff_r+max_0,:,:]
        lfb_adj[0:3,:,:]=0
        lfb_adj[-3:,:,:]=0
        
    if diff_c >= 0:
        hm_ll = hm_ll[:,diff_c:diff_c+cll,:]
    else:
        diff_c = -diff_c
        lfd_adj = lfd_adj[:,diff_c:diff_c+max_1,:]
        lfb_adj = lfb_adj[:,diff_c:diff_c+max_1,:]
        lfb_adj[:,0:3,:]=0
        lfb_adj[:,-3:,:]=0
        
    assert(lfd_adj.shape[0]==hm_ll.shape[0])
    assert(lfd_adj.shape[1]==hm_ll.shape[1])
    
    for vc_i in range(max_2):
        ri, ci = np.where(np.logical_and(lfb_adj[:,:,vc_i]==1,hm_ll[:,:,vc_i]>0))
        if ri.size==0:
            continue
            
        vc_tplt = vc_templates[vc_i]
        
        det = []
        for pp in zip(ri,ci):
            pp_dist = lfd_adj[pp[0]-3:pp[0]+4,pp[1]-3:pp[1]+4, vc_i]
            if pp_dist.shape[0]!=7 or pp_dist.shape[1]!=7:
                print(pp_dist.shape, lfd_adj.shape, hm_ll.shape, lfd.shape, heat_map.shape, pp)
            
            score1 = np.log(hm_ll[pp[0],pp[1],vc_i]) - hm_BG_val
            score2 = (1.7*49-np.sum(pp_dist**2))/24.5
            score3 = (np.sum((pp_dist - np.ones((7,7))*1.3)**2)-np.sum((pp_dist - vc_tplt)**2))/24.5
                
            det.append(score1+score2+score3)
            
        det = np.array(det).reshape(-1,1)
        nms_list = nms_help(ri, ci, det)
        
        total_score += np.sum(det[nms_list])
        
    return total_score


def get_scoreM(lfd, lfb, heat_map_M, vc_templates_M):
    rst = []
    for hm, vcT in zip(heat_map_M, vc_templates_M):
        rst.append(get_score(lfd, lfb, hm, vcT))
        
    return np.max(rst)


def nms_help(ri, ci, det, Astride=16, Arf=100, Apad=42, nms_thrh = 0.05):
    r_list = Astride * ri + Arf/2 - Apad
    c_list = Astride * ci + Arf/2 - Apad
    bb_loc = np.column_stack([c_list-Arf/2, r_list-Arf/2, c_list+Arf/2, r_list+Arf/2])
    bb_loc_ = np.column_stack([bb_loc, det])
    nms_list = nms(bb_loc_, nms_thrh)
    
    return nms_list


def get_encoding(lfd, magic_thrh):
    lfb = lfd < magic_thrh
    lfb[0:3, :, :] = 0
    lfb[-3:, :, :] = 0
    lfb[:, 0:3, :] = 0
    lfb[:, -3:, :] = 0
    
    return lfb