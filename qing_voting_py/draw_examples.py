import pickle
import numpy as np
import cv2

# fname='/mnt/4T-HD/qing/intermediate/feat/all_mergelist_rand_both_example.pickle'
fname='/export/home/qliu24/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_ALEX_conv3_K200_gray_example.pickle'
with open(fname,'rb') as fh:
    example = pickle.load(fh)

print(len(example))

for ii in range(len(example)):
    big_img = np.zeros((10+(100+10)*4, 10+(100+10)*5, 3))
    for iis in range(20):
        if iis >= example[ii].shape[1]:
            continue

        aa = iis//5
        bb = iis%5
        rnum = 10+aa*(100+10)
        cnum = 10+bb*(100+10)
        big_img[rnum:rnum+99, cnum:cnum+99, :] = example[ii][:,iis].reshape(99,99,3).astype(int)

    fname = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/examples_K200_gray_alex_nocat/example_K' + str(ii) + '.png'
    cv2.imwrite(fname, big_img)
