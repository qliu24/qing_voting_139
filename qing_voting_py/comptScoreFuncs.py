import math
import numpy as np
from nms import nms

def get_score(lfd, lfb, heat_map, vc_templates):
    max_0, max_1 = heat_map[0][0].shape
    max_2 = len(heat_map)
    heat_map_min = 1/(max_0*max_1)
    hm_BG_val = np.log(heat_map_min)
    
    total_score = 0
    
    rll,cll = lfd.shape[0:2]
    diff_r = np.floor((max_0-rll)/2).astype(int)
    diff_c = np.floor((max_1-cll)/2).astype(int)
    if diff_r >= 0:
        hm_ll = [[heat_map[vc_i][hmnn][diff_r:diff_r+rll,:] for hmnn in range(len(heat_map[vc_i]))] for vc_i in range(max_2)]
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
        hm_ll = [[hm_ll[vc_i][hmnn][:,diff_c:diff_c+cll] for hmnn in range(len(heat_map[vc_i]))] for vc_i in range(max_2)]
    else:
        diff_c = -diff_c
        lfd_adj = lfd_adj[:,diff_c:diff_c+max_1,:]
        lfb_adj = lfb_adj[:,diff_c:diff_c+max_1,:]
        lfb_adj[:,0:3,:]=0
        lfb_adj[:,-3:,:]=0
        
    
    for vc_i in range(max_2):
        for hmnn in range(len(heat_map[vc_i])):
            ri, ci = np.where(np.logical_and(lfb_adj[:,:,vc_i]==1, hm_ll[vc_i][hmnn]>0))
            if ri.size==0:
                continue
            
            vc_tplt = vc_templates[vc_i][hmnn]
            
            det = []
            for pp in zip(ri,ci):
                pp_dist = lfd_adj[pp[0]-3:pp[0]+4,pp[1]-3:pp[1]+4, vc_i]
                
                score1 = np.log(hm_ll[vc_i][hmnn][pp[0],pp[1]]) - hm_BG_val
                score2 = (1.7*49-np.sum(pp_dist**2))/10
                score3 = (np.sum((pp_dist - np.ones((7,7))*1.3)**2)-np.sum((pp_dist - vc_tplt)**2))/5
                
                det.append(score1+score2+score3)
            
            total_score += np.max(det)
        
    return total_score


def get_scoreM(lfd, lfb, heat_map_M, vc_templates_M):
    rst = []
    for hm, vcT in zip(heat_map_M, vc_templates_M):
        rst.append(get_score(lfd, lfb, hm, vcT))
        
    return np.max(rst)


def get_encoding(lfd, magic_thrh):
    lfb = lfd < magic_thrh
    lfb[0:3, :, :] = 0
    lfb[-3:, :, :] = 0
    lfb[:, 0:3, :] = 0
    lfb[:, -3:, :] = 0
    
    return lfb