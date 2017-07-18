import numpy as np

def knn_cls_no_neg(K, mat_dis, N_f, N_r, real_label, coef_std = 2):
    N = mat_dis.shape[0]
    assert(N==N_r+N_f)
    np.fill_diagonal(mat_dis, 9999)
    real_dis_ls = np.array([])
    real_mat = mat_dis[N_f:N, N_f:N]
    for nr in range(N_r):
        idx_sort = np.argsort(real_mat[nr])[0:5]
        real_dis_ls = np.append(real_dis_ls, real_mat[nr, idx_sort])
        
    thrh = np.mean(real_dis_ls) + coef_std*np.std(real_dis_ls)
    
    rst = np.zeros(N)
    mat_dis_obs = mat_dis[:, N_f:N]
    for nn in range(N):
        nn_idx = np.argsort(mat_dis_obs[nn])[0:K]
        if np.sum(mat_dis_obs[nn, nn_idx]<thrh)/K < 0.5:
            rst[nn] = 1-real_label
        else:
            rst[nn] = real_label
        
    return rst

def get_stats(dat):
    N = dat.shape[0]
    dim1,dim2 = dat.shape[1:3]
    fire_cnt = np.sum(dat, axis=3)
    fire_cnt_per_sample = [np.mean(fire_cnt[nn]) for nn in range(N)]
    stat1 = [np.mean(fire_cnt_per_sample), np.std(fire_cnt_per_sample)]
    
    no_cover_per_sample = [np.sum(fire_cnt[nn]==0)/(dim1*dim2) for nn in range(N)]
    stat2 = [np.mean(no_cover_per_sample), np.std(no_cover_per_sample)]
    return stat1, stat2
    
    
def normalize_features(features):
    '''features: n by d matrix'''
    assert(len(features.shape)==2)
    return features/np.sqrt(np.sum(features**2, axis=1).reshape(-1,1))