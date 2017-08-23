import pickle
import numpy as np
import cv2
import os

category = 'train'
fname='/export/home/qliu24/qing_voting_139/qing_voting_py/data/dict/dictionary_PASCAL3D+_{}_VGG16_pool4_K150_vMFMM30_example.pickle'.format(category)

with open(fname,'rb') as fh:
    example = pickle.load(fh)

print(len(example))

savedir = '/export/home/qliu24/qing_voting_139/qing_voting_py/data/dict/examples_K{}_{}/'.format(len(example), category)

if not os.path.exists(savedir):
    os.makedirs(savedir)

Arf = 100

for ii in range(len(example)):
    big_img = np.zeros((10+(Arf+10)*4, 10+(Arf+10)*5, 3))
    for iis in range(20):
        if iis >= example[ii].shape[1]:
            continue

        aa = iis//5
        bb = iis%5
        rnum = 10+aa*(Arf+10)
        cnum = 10+bb*(Arf+10)
        big_img[rnum:rnum+Arf, cnum:cnum+Arf, :] = example[ii][:,iis].reshape(Arf,Arf,3).astype(int)

    fname = savedir + 'example_K' + str(ii) + '.png'
    cv2.imwrite(fname, big_img)
