import numpy as np
import random
import scipy.misc
import json
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
import pickle
from utils import *

file_path = '/export/home/qliu24/VC_adv_data/feng/'

for vc in range(0,70,10):
    fake_dic = json.load(open(file_path+ 'feng_vc' + str(vc) + '_fake', 'r'))
    fake_code4 = np.array(fake_dic['fake_code4'])
    fake_code3 = np.array(fake_dic['fake_code3'])
    print('Target VC: {}'.format(vc))
    print('Pool3 stats:')
    print(get_stats(fake_code3))

    ori_dic = json.load(open(file_path+ 'feng_vc' + str(vc) + '_fake_ori', 'r'))
    ori4 = np.array(ori_dic['ori4'])
    ori_dis4 = np.array(ori_dic['ori_dis4'])
    ori_code4 = np.array(ori_dic['ori_code4'])

    real_dic = json.load(open(file_path+ 'vc' + str(vc) + '_real', 'r'))
    real_code4 = np.array(real_dic['real_code4'])
    real_code3 = np.array(real_dic['real_code3'])

    msk_fake = np.logical_and(np.logical_not(ori_code4[:,vc]), fake_code4[:,vc])
    print('{0} successful adv out of {1}'.format(np.sum(msk_fake), fake_code4.shape[0]))
    msk_real = real_code4[:,vc]
    print('{0} true samples out of {1}'.format(np.sum(msk_real), real_code4.shape[0]))
    msk = np.concatenate([msk_fake, msk_real], axis=0)
    concat = np.concatenate((fake_code3, real_code3), axis=0)
    concat = concat[msk]
    
    fname = file_path + 'simmat_vc{}.pickle'.format(vc)
    with open(fname, 'rb') as fh:
        mat1, mat2 = pickle.load(fh)

    mat = mat1
    N = mat.shape[0]
    assert(N == concat.shape[0])
    mat_full = mat + mat.T - np.ones((N,N))
    np.fill_diagonal(mat_full, 0)
    
    N_f = np.sum(msk_fake)
    N_r = np.sum(msk_real)
    label1 = np.concatenate((np.zeros((N_f, )), np.ones((N_r, ))), axis=0)
    label2 = np.concatenate((np.ones((N_f, )), np.zeros((N_r, ))), axis=0)

    kk=5
    predict1 = knn_cls_no_neg(kk, mat_full, N_f, N_r, 1, coef_std=2)
    predict2 = knn_cls_no_neg(kk, mat_full, N_f, N_r, 0, coef_std=2)
    print('all set result:')
    print("{0},{1},{2}".format(f1_score(label1, predict1), f1_score(label2, predict2), np.sum(label1[0:N_f] == predict1[0:N_f])/N_f))
    
    ori_dis_vc = ori_dis4[:,vc][msk_fake]
    pctl0 = 0
    pctl1 = np.percentile(ori_dis_vc, 25)
    pctl2 = np.percentile(ori_dis_vc, 50)
    pctl3 = np.percentile(ori_dis_vc, 75)
    pctl4 = 1.0
    print('percentiles:')
    print(pctl1, pctl2, pctl3)
    
    pctl1= 0.6
    pctl2 = 0.7
    pctl3 = 0.8

    pctl_ls = [pctl0, pctl1, pctl2, pctl3, pctl4]
    for p1,p2 in zip(pctl_ls[0:-1], pctl_ls[1:]):
        msk_pctl = np.logical_and(ori_dis_vc>=p1, ori_dis_vc<p2)
        N_f2 = np.sum(msk_pctl)
        msk_concat = np.concatenate([msk_pctl, np.ones(N_r).astype(bool)], axis=0)
        label1_pctl = label1[msk_concat]
        label2_pctl = label2[msk_concat]
        mat_full_pctl = mat_full[msk_concat][:,msk_concat]
        predict1 = knn_cls_no_neg(kk, mat_full_pctl, N_f2, N_r, 1, coef_std=2)
        predict2 = knn_cls_no_neg(kk, mat_full_pctl, N_f2, N_r, 0, coef_std=2)
        print("{0},{1},{2}".format(f1_score(label1_pctl, predict1), f1_score(label2_pctl, predict2), np.sum(label1_pctl[0:N_f2] == predict1[0:N_f2])/N_f2))