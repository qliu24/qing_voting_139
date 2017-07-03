import pickle
import numpy as np
from scipy.spatial.distance import pdist
from copy import *
from sklearn.mixture import GaussianMixture

def get_fired_pos_gm(layer_f_dist, layer_f_b, gm_ls, vc_part_cnt, max_0, max_1, max_2):
    heat_map_min = 1/(max_0*max_1)
    hm_BG_val = np.log(heat_map_min)
    
    layer_f_pos = [[[None for hmnn in range(vc_part_cnt[vc_i])] for vc_i in range(max_2)] for nn in range(len(layer_f_dist))]
    
    for lfd_i,lfd in enumerate(layer_f_dist):
        lfb = layer_f_b[lfd_i]
        rll,cll = lfd.shape[0:2]
        diff_r = int((max_0-rll)/2)
        diff_c = int((max_1-cll)/2)
        
        for vc_i in range(max_2):
            ri, ci = np.where(lfb[:,:,vc_i]==1)
            if ri.size==0:
                continue
            
            pos_lsi = np.column_stack([ri+diff_r, ci+diff_c])
            if vc_part_cnt[vc_i] == 1:
                ass = np.zeros(len(ri))
            elif vc_part_cnt[vc_i] > 1:
                ass = gm_ls[vc_i].predict(pos_lsi)
                
            for kk in range(vc_part_cnt[vc_i]):
                if np.sum(ass==kk) == 0:
                    continue

                det = []
                for pp in zip(ri[ass==kk],ci[ass==kk]):
                    pp_dist = lfd[pp[0]-3:pp[0]+4,pp[1]-3:pp[1]+4, vc_i]
                    score2 = (1.7*49-np.sum(pp_dist**2))/10
                    det.append(score2)

                layer_f_pos[lfd_i][vc_i][kk] = pos_lsi[ass==kk][np.argmax(det)]
            
    return layer_f_pos


def get_fired_pos_vct(layer_f_dist, layer_f_b, gm_ls, vc_templates, vc_part_cnt, max_0, max_1, max_2):
    heat_map_min = 1/(max_0*max_1)
    hm_BG_val = np.log(heat_map_min)
    
    layer_f_pos = [[[None for hmnn in range(vc_part_cnt[vc_i])] for vc_i in range(max_2)] for nn in range(len(layer_f_dist))]
    
    for lfd_i,lfd in enumerate(layer_f_dist):
        lfb = layer_f_b[lfd_i]
        rll,cll = lfd.shape[0:2]
        diff_r = int((max_0-rll)/2)
        diff_c = int((max_1-cll)/2)
        
        for vc_i in range(max_2):
            ri, ci = np.where(lfb[:,:,vc_i]==1)
            if ri.size==0:
                continue
            
            pos_lsi = np.column_stack([ri+diff_r, ci+diff_c])
            if vc_part_cnt[vc_i] == 1:
                ass = np.zeros(len(ri))
            elif vc_part_cnt[vc_i] > 1:
                ass = gm_ls[vc_i].predict(pos_lsi)
                
            for kk in range(vc_part_cnt[vc_i]):
                if np.sum(ass==kk) == 0:
                    continue

                vc_tplt = vc_templates[vc_i][kk]
                det = []
                for pp in zip(ri[ass==kk],ci[ass==kk]):
                    pp_dist = lfd[pp[0]-3:pp[0]+4,pp[1]-3:pp[1]+4, vc_i]
                    score2 = (1.7*49-np.sum(pp_dist**2))/10
                    score3 = (np.sum((pp_dist - np.ones((7,7))*1.3)**2)-np.sum((pp_dist - vc_tplt)**2))/5
                    det.append(score2+score3)

                layer_f_pos[lfd_i][vc_i][kk] = pos_lsi[ass==kk][np.argmax(det)]
            
    return layer_f_pos


def get_fired_pos(layer_f_dist, layer_f_b, heat_map, vc_part_cnt, vc_part_p, vc_templates):
    max_0, max_1 = heat_map[0][0].shape
    max_2 = len(heat_map)
    heat_map_min = 1/(max_0*max_1)
    hm_BG_val = np.log(heat_map_min)
    
    layer_f_pos = [[[None for hmnn in range(vc_part_cnt[vc_i])] for vc_i in range(max_2)] for nn in range(len(layer_f_dist))]
    
    for lfd_i,lfd in enumerate(layer_f_dist):
        lfb = layer_f_b[lfd_i]
        rll,cll = lfd.shape[0:2]
        diff_r = int((max_0-rll)/2)
        diff_c = int((max_1-cll)/2)
        
        for vc_i in range(max_2):
            for hmnn in range(vc_part_cnt[vc_i]):
                hm = heat_map[vc_i][hmnn]
                hm_ll = hm[diff_r:diff_r+rll,diff_c:diff_c+cll]
                
                vc_tplt = vc_templates[vc_i][hmnn]
                
                ri, ci = np.where(np.logical_and(lfb[:,:,vc_i]==1,hm_ll[:,:]>0))
                if ri.size==0:
                    continue
                
                pos_lsi = np.column_stack([ri+diff_r, ci+diff_c])
                
                det = []
                for pp in zip(ri, ci):
                    pp_dist = lfd[pp[0]-3:pp[0]+4,pp[1]-3:pp[1]+4, vc_i]
                    score1 = np.log(hm_ll[pp[0],pp[1]]) + np.log(vc_part_p[vc_i][hmnn]) - hm_BG_val
                    score2 = (1.7*49-np.sum(pp_dist**2))/10
                    score3 = (np.sum((pp_dist - np.ones((7,7))*1.3)**2)-np.sum((pp_dist - vc_tplt)**2))/5
                    det.append(score1+score2+score3)
                    
                layer_f_pos[lfd_i][vc_i][hmnn] = pos_lsi[np.argmax(det)]
                
    return layer_f_pos


def get_heatmap(layer_f_pos, max_0, max_1, max_2, blur=0.5, pctl = None, thrh = None):
    heat_map = [[np.zeros((max_0, max_1)) for hmnn in range(len(layer_f_pos[0][vc_i]))] for vc_i in range(max_2)]
    for vc_i in range(max_2):
        for hmnn in range(len(layer_f_pos[0][vc_i])):
            vc_p = []
            for lfp in layer_f_pos:
                if not(lfp[vc_i][hmnn] is None):
                    vc_p.append(lfp[vc_i][hmnn])
            
            if len(vc_p)==0:
                print('empty vc_p for {0}_{1}'.format(vc_i, hmnn))
                continue
                
            heat_map[vc_i][hmnn] = init_heatmap(vc_p, max_0, max_1, blur, pctl, thrh)
            
    return heat_map


def init_heatmap(vc_p, max_0, max_1, blur=0.5, pctl = None, thrh = None):
    hm = np.zeros((max_0, max_1))
    for pp in vc_p:
        if blur is None:
            hm[pp[0],pp[1]]+=1
        else:
            hm[pp[0],pp[1]]+=1.0-blur
            hm[pp[0]-1:pp[0]+2,pp[1]-1:pp[1]+2]+=blur
            
    if pctl is None and thrh is None:
        thrh = 0
    elif not (pctl is None):
        thrh = np.percentile(hm[hm>0], pctl)
        
    hm[hm<=thrh]=0
    hm /= np.sum(hm)
    return hm


def get_vctplt(layer_f_pos, layer_f_dist, vc_part_cnt, max_0, max_1, max_2):
    vc_templates = [[np.zeros((7,7,0)) for hmnn in range(vc_part_cnt[vc_i])] for vc_i in range(max_2)]
    
    for lfp,lfd in zip(layer_f_pos, layer_f_dist):
        rll,cll = lfd.shape[0:2]
        diff_r = int((max_0-rll)/2)
        diff_c = int((max_1-cll)/2)
        
        for vc_i in range(max_2):
            for hmnn in range(vc_part_cnt[vc_i]):
                if lfp[vc_i][hmnn] is None:
                    continue
                    
                ri,ci = lfp[vc_i][hmnn]
                rr = ri-diff_r
                cc = ci-diff_c
                patch_dist = lfd[rr-3:rr+4, cc-3:cc+4, vc_i].reshape(7,7,1)
                vc_templates[vc_i][hmnn] = np.concatenate([vc_templates[vc_i][hmnn], patch_dist], axis=2)
                
    for vc_i in range(max_2):
        for hmnn in range(vc_part_cnt[vc_i]):
            vc_templates[vc_i][hmnn] = np.median(vc_templates[vc_i][hmnn], axis=2)
        
    return vc_templates


def get_score(layer_f_dist, layer_f_pos, heat_map, vc_templates, vc_part_cnt, vc_part_p):
    max_0, max_1 = heat_map[0][0].shape
    max_2 = len(heat_map)
    heat_map_min = 1/(max_0*max_1)
    hm_BG_val = np.log(heat_map_min)
    
    total_score = 0
    total_score_details = np.zeros(3)
    for lfd, lfp in zip(layer_f_dist, layer_f_pos):
        rll, cll = lfd.shape[0:2]
        diff_r = int((max_0-rll)/2)
        diff_c = int((max_1-cll)/2)
        
        for vc_i in range(max_2):
        # for vc_i in [17,30,38,81,108]:
            for hmnn in range(vc_part_cnt[vc_i]):
                if lfp[vc_i][hmnn] is None:
                    continue
                
                if vc_part_p[vc_i][hmnn] < 0.1:
                    continue
                    
                rp = lfp[vc_i][hmnn][0] - diff_r
                cp = lfp[vc_i][hmnn][1] - diff_c
                pp_dist = lfd[rp-3:rp+4,cp-3:cp+4, vc_i]
                
                vc_tplt = vc_templates[vc_i][hmnn]
                
                score1 = np.log(heat_map[vc_i][hmnn][lfp[vc_i][hmnn][0],lfp[vc_i][hmnn][1]]) + np.log(vc_part_p[vc_i][hmnn]) - hm_BG_val
                score2 = (1.7*49-np.sum(pp_dist**2))/10
                score3 = (np.sum((pp_dist - np.ones((7,7))*1.3)**2)-np.sum((pp_dist - vc_tplt)**2))/5
                
                # print(vc_i,hmnn,score1,score2)
                
                total_score += score1+score2+score3
                total_score_details += np.array([score1,score2,score3])
                
    return total_score/len(layer_f_dist),total_score_details/len(layer_f_dist)
        

def gm_vc_pos(pos_ls, cluster_num):
    gm = GaussianMixture(n_components=cluster_num, covariance_type='full', \
                                  n_init=10, max_iter=1500)
    
    gm.fit(pos_ls)
    assignment = gm.predict(pos_ls)
    return gm, assignment


oo='car'
magic_thh = 0.85

file_path1 = '/export/home/qliu24/qing_voting_data/intermediate/feat_car/'
file_path2 = '/export/home/qliu24/qing_voting_data/intermediate/VCpart_model_car/'

fname = file_path1 + oo + '_mergelist_rand_train_carVC.pickle'
print('loading object {0}'.format(oo))
with open(fname, 'rb') as fh:
    _,layer_feature_dist = pickle.load(fh)

            
N = len(layer_feature_dist)
print('total number of instances {0}'.format(N))

fname = file_path2 + '{0}_k2_2_lbs.pickle'.format(oo)
with open(fname, 'rb') as fh:
    lbs = pickle.load(fh)

K=4
save_name = file_path2+'VCpart_model_{0}_K{1}_term23_noblur_coef.pickle'.format(oo, K)

hm_ls = []
vcpartp_ls = []
vcT_ls = []
for kk in range(K):
    idx_s = np.where(lbs==kk)[0]
    
    layer_feature_dist_kk = [layer_feature_dist[nn] for nn in idx_s]
    
    N = len(layer_feature_dist_kk)
    print('total number of instances {0} for cluster {1}'.format(N, kk))
    
    layer_feature_b = [None for nn in range(N)]
    for nn in range(N):
        lfb = (layer_feature_dist_kk[nn]<magic_thh).astype(int)
        lfb[0:3, :, :] = 0
        lfb[-3:, :, :] = 0
        lfb[:, 0:3, :] = 0
        lfb[:, -3:, :] = 0
        layer_feature_b[nn] = deepcopy(lfb)
    
    max_0 = max([lfb.shape[0] for lfb in layer_feature_b])
    max_1 = max([lfb.shape[1] for lfb in layer_feature_b])
    max_2 = max([lfb.shape[2] for lfb in layer_feature_b])
    
    print('heatmap shape: {0}'.format([max_0, max_1, max_2]))
    
    gm_ls = [None for vc_i in range(max_2)]
    vc_part_cnt = np.zeros(max_2).astype(int)
    for vc_i in range(max_2):
        if vc_i%20 == 0:
            print('VC idx: {0}'.format(vc_i))
        vc_p = []
        for nn in range(N):
            vc_l = layer_feature_b[nn][:,:,vc_i]
            rnum, colnum = vc_l.shape
            diffr = int((max_0-rnum)/2)
            diffc = int((max_1-colnum)/2)
            row_i, col_i = np.where(vc_l==1)
            for pp in zip(row_i,col_i):
                vc_p.append((pp[0]+diffr,pp[1]+diffc))

        if len(vc_p)<N/2:
            print('not enough VC firing {0}'.format(vc_i))
            continue

        vc_p=np.array(vc_p)
        K=4
        gm, assignment = gm_vc_pos(vc_p, K)
        while True:
            p_dist = pdist(gm.means_)
            if np.any(gm.weights_<0.5/K) or np.any(p_dist<7):
                K -= 1
                if K==1:
                    gm = None
                    assignment = np.zeros(vc_p.shape[0])
                    break
                else:
                    gm, assignment = gm_vc_pos(vc_p, K)
            else:
                break

        # print('final K: {0}'.format(K), flush='True')
        gm_ls[vc_i]=gm
        vc_part_cnt[vc_i] = K
        
    
    layer_fired_pos = get_fired_pos_gm(layer_feature_dist_kk, layer_feature_b, gm_ls, vc_part_cnt, max_0, max_1, max_2)
    vc_templates = get_vctplt(layer_fired_pos, layer_feature_dist_kk, vc_part_cnt, max_0, max_1, max_2)
    layer_fired_pos = get_fired_pos_vct(layer_feature_dist_kk, layer_feature_b, gm_ls, vc_templates, vc_part_cnt, max_0, max_1, max_2)
    
    for itt in range(15):
        heat_map = get_heatmap(layer_fired_pos, max_0, max_1, max_2, blur=None, pctl = None, thrh = None)
        vc_part_p = [[np.sum([not(layer_fired_pos[nn][vc_i][hmnn] is None) for nn in range(N)])/N for hmnn in range(vc_part_cnt[vc_i])]for vc_i in range(max_2)]
        sc, sc_d = get_score(layer_feature_dist_kk, layer_fired_pos, heat_map, vc_templates, vc_part_cnt, vc_part_p)
        print(sc, sc_d)
        
        vc_templates = get_vctplt(layer_fired_pos, layer_feature_dist_kk, vc_part_cnt, max_0, max_1, max_2)
        layer_fired_pos = get_fired_pos(layer_feature_dist_kk, layer_feature_b, heat_map, vc_part_cnt, vc_part_p, vc_templates)
    
    
    heat_map = get_heatmap(layer_fired_pos, max_0, max_1, max_2, blur=0.25, pctl = None, thrh = None)
    vc_part_p = [[np.sum([not(layer_fired_pos[nn][vc_i][hmnn] is None) for nn in range(N)])/N for hmnn in range(vc_part_cnt[vc_i])] for vc_i in range(max_2)]
    sc, sc_d = get_score(layer_feature_dist_kk, layer_fired_pos, heat_map, vc_templates, vc_part_cnt, vc_part_p)
    print(sc, sc_d)
    
    hm_ls.append(heat_map)
    vcpartp_ls.append(vc_part_p)
    vcT_ls.append(vc_templates)
    
with open(save_name, 'wb') as fh:
    pickle.dump([hm_ls, vcpartp_ls, vcT_ls], fh)
    
        





