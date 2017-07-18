from joblib import Parallel, delayed
import pickle
import numpy as np
import math
from vcdist_funcs import *
import time
import json

paral_num=20
file_path = '/export/home/qliu24/VC_adv_data/feng/'
for vc in range(0,200,10):
    print('VC: {0}'.format(vc))
    savename = file_path + 'simmat_vc{0}.pickle'.format(vc)
    
    fake_dic = json.load(open(file_path+ 'feng_vc' + str(vc) + '_fake', 'r'))
    fake_code4 = np.array(fake_dic['fake_code4'])
    fake_code3 = np.array(fake_dic['fake_code3'])

    ori_dic = json.load(open(file_path+ 'feng_vc' + str(vc) + '_fake_ori', 'r'))
    ori4 = np.array(ori_dic['ori4'])
    ori_dis4 = np.array(ori_dic['ori_dis4'])
    ori_code4 = np.array(ori_dic['ori_code4'])

    real_dic = json.load(open(file_path+ 'vc' + str(vc) + '_real', 'r'))
    real_code4 = np.array(real_dic['real_code4'])
    real_code3 = np.array(real_dic['real_code3'])

    msk_fake = np.logical_and(np.logical_not(ori_code4[:,vc]), fake_code4[:,vc])
    msk_real = real_code4[:,vc]
    msk = np.concatenate([msk_fake, msk_real], axis=0)
    concat = np.concatenate((fake_code3, real_code3), axis=0)
    concat = concat[msk]
    
    '''
    fake_dic = json.load(open(file_path+ 'vc' + str(vc) + 'fake', 'r'))
    real_dic = json.load(open(file_path+ 'vc' + str(vc) + 'real', 'r'))
    fake_code3 = np.array(fake_dic['fake_code3'])
    real_code3 = np.array(real_dic['real_code3'])
    fake_code3 = np.delete(fake_code3, (vc, ), axis=0)
    concat = np.concatenate((fake_code3, real_code3), axis=0)
    label = np.concatenate((np.zeros((fake_code3.shape[0], )), np.ones((real_code3.shape[0], ))), axis=0)
    '''
    N = concat.shape[0]

    print('total number of instances {0}'.format(N))
    print('concat shape: {}'.format(concat.shape))

    print('Start compute sim matrix...')
    _s = time.time()

    mat_dis1 = np.ones((N,N))
    mat_dis2 = np.ones((N,N))
    for nn1 in range(N-1):
        if nn1%50 == 0:
            print(nn1, end=' ', flush=True)
        
        # inputs = [(concat[nn1], concat[nn2]) for nn2 in range(nn1+1,N)]
        # mat_dis[nn1, nn1+1:N] = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dist_rigid_transfer_sym)(i) for i in inputs))
        inputs = [(concat[nn1].T, concat[nn2].T) for nn2 in range(nn1+1,N)]
        para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_sym2)(i) for i in inputs))
        mat_dis1[nn1, nn1+1:N] = para_rst[:,0]
        mat_dis2[nn1, nn1+1:N] = para_rst[:,1]
        
        '''
        for nn2 in range(nn1+1, N):
            if nn2%100 == 0:
                print(nn2, end=' ', flush=True)
            
            mat_dis[nn1, nn2] = vc_dist_rigid_transfer_sym([concat[nn1], concat[nn2]])
        '''

    _e = time.time()
    print((_e-_s)/60)
    print(mat_dis1.shape)

    with open(savename, 'wb') as fh:
        pickle.dump([mat_dis1, mat_dis2], fh)


