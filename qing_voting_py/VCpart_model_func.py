import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from nms import nms
from copy import *


def get_fired_pos(layer_f_dist, layer_f_b, heat_map, vc_templates=None):
    max_0, max_1 = heat_map[0][0].shape
    max_2 = len(heat_map)
    heat_map_min = 1/(max_0*max_1)
    hm_BG_val = np.log(heat_map_min)
    
    total_score = 0
    total_score_details = np.zeros(3)
    layer_f_pos = [[[None for hmnn in range(len(heat_map[vc_i]))] for vc_i in range(max_2)] for lfd in layer_f_dist]
    
    for lfd_i,lfd in enumerate(layer_f_dist):
        lfb = layer_f_b[lfd_i]
        rll,cll = lfd.shape[0:2]
        diff_r = int((max_0-rll)/2)
        diff_c = int((max_1-cll)/2)
        
        for vc_i in range(max_2):
            for hmnn,hm in enumerate(heat_map[vc_i]):
                hm_ll = hm[diff_r:diff_r+rll,diff_c:diff_c+cll]
                
                ri, ci = np.where(np.logical_and(lfb[:,:,vc_i]==1,hm_ll[:,:]>0))
                if ri.size==0:
                    continue
                
                if vc_templates is None:
                    vc_tplt = np.ones((7,7))*1.3
                else:
                    vc_tplt = vc_templates[vc_i][hmnn]
                
                det = []
                det_details = []
                for pp in zip(ri,ci):
                    pp_dist = lfd[pp[0]-3:pp[0]+4,pp[1]-3:pp[1]+4, vc_i]
                    score1 = np.log(hm_ll[pp[0],pp[1]]) - hm_BG_val
                    score2 = (1.7*49-np.sum(pp_dist**2))/10
                    score3 = (np.sum((pp_dist - np.ones((7,7))*1.3)**2)-np.sum((pp_dist - vc_tplt)**2))/5
                    det.append(score1+score2+score3)
                    det_details.append([score1,score2,score3])
                    
                max_idx = np.argmax(det)
                
                layer_f_pos[lfd_i][vc_i][hmnn] = (ri[max_idx]+diff_r, ci[max_idx]+diff_c)
                total_score += det[max_idx]
                total_score_details += np.array(det_details[max_idx])
            
    return layer_f_pos, total_score/len(layer_f_dist), total_score_details/len(layer_f_dist)


def get_heatmap(layer_f_pos, max_0, max_1, max_2, pctl=None, blur=True):
    heat_map = [[np.zeros((max_0, max_1)) for hmnn in range(len(layer_f_pos[0][vc_i]))] for vc_i in range(max_2)]
    for vc_i in range(max_2):
        for hmnn in range(len(layer_f_pos[0][vc_i])):
            vc_p = []
            for lfp in layer_f_pos:
                if lfp[vc_i][hmnn] is not None:
                    vc_p.append(lfp[vc_i][hmnn])
            
            if len(vc_p)==0:
                print('empty vc_p for {0}_{1}'.format(vc_i, hmnn))
                continue
                
            heat_map[vc_i][hmnn] = init_heatmap(vc_p, max_0, max_1, pctl, blur)
            
    return heat_map

    
def get_vctplt(layer_f_pos, layer_f_dist, max_0, max_1):
    vc_templates = [[np.zeros((7,7,0)) for hmnn in range(len(layer_f_pos[0][vc_i]))] for vc_i in range(len(layer_f_pos[0]))]
    
    for lfp,lfd in zip(layer_f_pos, layer_f_dist):
        rll,cll = lfd.shape[0:2]
        diff_r = int((max_0-rll)/2)
        diff_c = int((max_1-cll)/2)
        
        for vc_i in range(lfd.shape[2]):
            for hmnn in range(len(lfp[vc_i])):
                if lfp[vc_i][hmnn] is None:
                    continue
                    
                ri,ci = lfp[vc_i][hmnn]
                rr = ri-diff_r
                cc = ci-diff_c
                patch_dist = lfd[rr-3:rr+4, cc-3:cc+4, vc_i].reshape(7,7,1)
                vc_templates[vc_i][hmnn] = np.concatenate([vc_templates[vc_i][hmnn], patch_dist], axis=2)
                
    for vc_i in range(layer_f_dist[0].shape[2]):
        for hmnn in range(len(layer_f_pos[0][vc_i])):
            vc_templates[vc_i][hmnn] = np.median(vc_templates[vc_i][hmnn], axis=2)
        
    return vc_templates


def init_heatmap(vc_p, max_0, max_1, pctl = None, blur=True):
    hm = np.zeros((max_0, max_1))
    for pp in vc_p:
        if blur:
            hm[pp[0],pp[1]]+=0.5
            hm[pp[0]-1:pp[0]+2,pp[1]-1:pp[1]+2]+=0.5
        else:
            hm[pp[0],pp[1]]+=1
        
    if pctl is None:
        thrh = 0
    else:
        thrh = np.percentile(hm[hm>0], pctl)
        
    hm[hm<thrh]=0
    hm /= np.sum(hm)
    return hm


def gm_vc_pos(pos_ls, cluster_num):
    gm = GaussianMixture(n_components=cluster_num, covariance_type='full', \
                                  n_init=10, max_iter=1500)
    
    gm.fit(pos_ls)
    assignment = gm.predict(pos_ls)
    return gm, assignment

'''

def get_nms(layer_f_dist, layer_f_b, heat_map, vc_templates=None):
    heat_map_min = 1/350
    hm_BG_val = np.log(heat_map_min)
    
    total_score = 0
    total_score_details = np.zeros(3)
    layer_f_nms = [np.zeros_like(lfd) for lfd in layer_f_dist]
    max_0, max_1, max_2 = heat_map.shape
    for lfd_i,lfd in enumerate(layer_f_dist):
        lfb = layer_f_b[lfd_i]
        rll,cll = lfd.shape[0:2]
        diff_r = int((max_0-rll)/2)
        diff_c = int((max_1-cll)/2)
        hm_ll = heat_map[diff_r:diff_r+rll,diff_c:diff_c+cll,:]
        
        for vc_i in range(max_2):
            ri, ci = np.where(np.logical_and(lfb[:,:,vc_i]==1,hm_ll[:,:,vc_i]>0))
            if ri.size==0:
                continue
            
            if vc_templates is None:
                vc_tplt = np.zeros((7,7))
            else:
                vc_tplt = vc_templates[vc_i]
            
            det = []
            det_details = []
            for pp in zip(ri,ci):
                pp_dist = lfd[pp[0]-3:pp[0]+4,pp[1]-3:pp[1]+4, vc_i]
                score1 = np.log(hm_ll[pp[0],pp[1],vc_i]) - hm_BG_val
                score2 = (1.7*49-np.sum(pp_dist**2))/24.5
                score3 = (np.sum((pp_dist - np.ones((7,7))*1.3)**2)-np.sum((pp_dist - vc_tplt)**2))/24.5
                det.append(score1+score2+score3)
                det_details.append([score1,score2,score3])
                
            det = np.array(det).reshape(-1,1)
            det_details = np.array(det_details)
            nms_list = nms_help(ri, ci, det)
            
            layer_f_nms[lfd_i][ri[nms_list], ci[nms_list], vc_i] = 1
            total_score += np.sum(det[nms_list])
            total_score_details += np.sum(det_details[nms_list], axis=0)
            
    return layer_f_nms,total_score/len(layer_f_dist), total_score_details/len(layer_f_dist)
    

def nms_help(ri, ci, det, Astride=16, Arf=100, Apad=42):
    r_list = Astride * ri + Arf/2 - Apad
    c_list = Astride * ci + Arf/2 - Apad
    bb_loc = np.column_stack([c_list-Arf/2, r_list-Arf/2, c_list+Arf/2, r_list+Arf/2])
    bb_loc_ = np.column_stack([bb_loc, det])
    nms_list = nms(bb_loc_, 0.05)
    
    return nms_list


def get_heatmap(layer_f_nms, max_0, max_1, max_2):
    heat_map = np.zeros((max_0,max_1,max_2))
    for ni in range(len(layer_f_nms)):
        ri,ci,chi = layer_f_nms[ni].shape
        assert(chi==max_2)
        diff_r_1 = int((max_0-ri)/2)
        diff_r_2 = max_0-ri-diff_r_1
        
        diff_c_1 = int((max_1-ci)/2)
        diff_c_2 = max_1-ci-diff_c_1
        
        heat_map += np.pad(layer_f_nms[ni], ((diff_r_1, diff_r_2),(diff_c_1, diff_c_2),(0,0)), 'constant', constant_values=0)
        
    heat_map = heat_map/len(layer_f_nms)
    
    return heat_map


def get_blur_heatmap(layer_f_nms, max_0, max_1, max_2):
    heat_map = np.zeros((max_0,max_1,max_2))
    for ni in range(len(layer_f_nms)):
        ri,ci,chi = layer_f_nms[ni].shape
        assert(chi==max_2)
        diff_r_1 = int((max_0-ri)/2)
        diff_r_2 = max_0-ri-diff_r_1
        
        diff_c_1 = int((max_1-ci)/2)
        diff_c_2 = max_1-ci-diff_c_1
        
        padded = np.pad(layer_f_nms[ni], ((diff_r_1, diff_r_2),(diff_c_1, diff_c_2),(0,0)), 'constant', constant_values=0)
        for vc_i in range(padded.shape[2]):
            rll, cll = np.where(padded[:,:,vc_i]>0)
            for pp in zip(rll, cll):
                padded[max(0,pp[0]-1):min(padded.shape[0],pp[0]+2), max(0,pp[1]-1):min(padded.shape[1],pp[1]+2), vc_i]=1
            
        heat_map += padded
        
    # heat_map = heat_map/len(layer_f_nms)
    return heat_map
    
    
def get_vctplt(layer_f_nms, layer_f_dist):
    vc_templates = [np.zeros((7,7,0)) for vc_i in range(layer_f_dist[0].shape[2])]
    for lfnms,lfd in zip(layer_f_nms, layer_f_dist):
        for vc_i in range(lfd.shape[2]):
            ri,ci = np.where(lfnms[:,:,vc_i]==1)
            for pp in zip(ri,ci):
                patch_dist = lfd[pp[0]-3:pp[0]+4,pp[1]-3:pp[1]+4, vc_i].reshape(7,7,1)
                vc_templates[vc_i] = np.concatenate([vc_templates[vc_i], patch_dist], axis=2)
                
    for vc_i in range(layer_f_dist[0].shape[2]):
        vc_templates[vc_i] = np.median(vc_templates[vc_i], axis=2)
        
    return vc_templates
    

def trim_heatmap(heat_map, thrh_p = 25):
    heat_map_new = np.zeros_like(heat_map)
    for vc_i in range(heat_map.shape[2]):
        hm_vc = deepcopy(heat_map[:,:,vc_i])
        # thrh = np.mean(hm_vc[hm_vc>0])
        thrh = np.percentile(hm_vc[hm_vc>0], thrh_p)
        hm_vc[hm_vc<thrh]=0
        heat_map_new[:,:,vc_i] = hm_vc
        
    return np.array(heat_map_new)

def trim_heatmap2(heat_map):
    heat_map_new = np.zeros_like(heat_map)
    for vc_i in range(heat_map.shape[2]):
        hm_vc = deepcopy(heat_map[:,:,vc_i])
        thrh = 2.0/118
        hm_vc[hm_vc<thrh]=0
        heat_map_new[:,:,vc_i] = hm_vc
        
    return np.array(heat_map_new)

def blur_heatmap(heat_map):
    heat_map_new = np.zeros_like(heat_map)
    for vc_i in range(heat_map.shape[2]):
        hm_vc = deepcopy(heat_map[:,:,vc_i])
        rhm, chm = np.where(hm_vc>0)
        for pp in zip(rhm, chm):
            heat_map_new[max(0,pp[0]-1):min(hm_vc.shape[0],pp[0]+2), max(0,pp[1]-1):min(hm_vc.shape[1],pp[1]+2), vc_i] = \
            np.maximum(heat_map_new[max(0,pp[0]-1):min(hm_vc.shape[0],pp[0]+2), max(0,pp[1]-1):min(hm_vc.shape[1],pp[1]+2), vc_i], hm_vc[pp[0], pp[1]])
        
    return np.array(heat_map_new)


def normalize_heatmap(heat_map):
    heat_map_new = deepcopy(heat_map)
    for vc_i in range(heat_map_new.shape[2]):
        heat_map_new[:,:,vc_i] /= np.sum(heat_map_new[:,:,vc_i])
        
    return heat_map_new
    
'''