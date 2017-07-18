import numpy as np
import scipy.io as sio
import os
import cv2
import scipy.misc
from FeatureExtractor import FeatureExtractor
import h5py
import json
import pickle

f = open('dictionary_PASCAL3D+_car_VGG16_pool3_K200_vMFMM30.pickle', 'rb')
w = pickle.load(f)
sio.savemat('dictionary_PASCAL3D+_car_VGG16_pool3_K200_vMFMM30.mat', {'centers': w[1]})

vgg_mean = np.float32([[[103.939, 116.779, 123.68]]])
zero_mean = np.float32([[[0.0, 0.0, 0.0]]])
wrong_mean = np.float32([[[123.68, 116.779, 103.939]]])
cihang_mean_RGB = np.float32([[[122.7717, 115.9465, 102.9801]]])

pool3_file = 'dictionary_imagenet_car_vgg16_pool3_K223_norm_nowarp_prune_512.mat'
f3 = h5py.File(pool3_file)
centers3 = np.array(f3['centers'].value)

# pool4_file = 'dictionary_imagenet_car_vgg16_pool4_K176_norm_nowarp_prune_512.mat'
# f4 = h5py.File(pool4_file)
# centers4 = np.array(f4['centers'].value)

cihang_file = './centers.mat'
cihang_mat = sio.loadmat(cihang_file)
centers4 = np.transpose(cihang_mat['centers'])

fake_file = '../from_feng/constrained_multiple_adversial.mat'
ff = h5py.File(fake_file)

use_feng = False
fake = True
real = False
thres = 0.8
cache_dir = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'
extractor = FeatureExtractor(cache_folder = cache_dir, layer_names = ['pool3/MaxPool:0', 'pool4/MaxPool:0'])

for vc in range(1, 2):
    # Fake
    if fake:
        if use_feng:
            images = ff[ff['multiple_adversial'][vc][0]].value
            images = np.transpose(images, [0, 2, 3, 1])  # same as MATLAB, BGR order
            images = images[:, :, :, [2, 1, 0]]  # to RGB
        else:
            image_list = []
            vc_ori_list = []
            for n in range(1, 2000):
                file = sio.loadmat('../adv_patches/' + str(n * 5) + '.mat')
                image = np.array(file['im_fool'], dtype=np.float32)
                vc_ori = np.array(file['vc_idx_ori'], dtype=np.int32)
                vc_ori_list.append(vc_ori)
                image = np.expand_dims(image, axis=0)
                image_list.append(image)
                
            vc_ori = np.concatenate(vc_ori_list)
            images = np.concatenate(image_list, axis=0)
            images -= cihang_mean_RGB  # RGB here
        # images += cihang_mean_RGB
        # for i in range(3000):
        #     cv2.imshow('name', images[i, :, :, ::-1] / 255.0)
        #     cv2.waitKey(0)

        features, _ = extractor.extract_from_images(images)
        pool4 = features[1][:, :, :, :]
        normed4 = pool4 / np.linalg.norm(pool4, axis=-1).reshape((images.shape[0], 1, 1, 1))
        fake4 = normed4
        
        pool3 = features[0][:, 1:7, 1:7, :]
        normed3 = pool3 / np.linalg.norm(pool3, axis=-1).reshape((images.shape[0], 6, 6, 1))
        fake3 = normed3

        fake_dis4 = []
        for idx4 in range(176):
            vc4 = centers4[idx4, :]
            fake_temp4 = np.linalg.norm(fake4 - vc4, axis=-1)
            fake_dis4.append(fake_temp4)
            
        fake_dis4 = np.stack(fake_dis4, axis=-1)

        fake_dis3 = []
        for idx3 in range(223):
            vc3 = centers3[idx3, :]
            fake_temp3 = np.linalg.norm(np.abs(fake3 - vc3), axis=-1)
            fake_dis3.append(fake_temp3)
        fake_dis3 = np.stack(fake_dis3, axis=-1)

        fake_code4 = fake_dis4 < thres
        fake_code3 = fake_dis3 < thres

        temp = json.dumps({'fake_code4': fake_code4.tolist(), 'fake_code3': fake_code3.tolist()})
        if use_feng:
            target = open('vc' + str(vc) + 'fake', 'w')
        else:
            target = open('cihang_vc' + str(vc) + 'fake', 'w')

        target.write(temp)
        target.close()
    # Real
    if real:
        print(f4[f4['example'][vc][0]].shape)
        images = np.array(f4[f4['example'][vc][0]], dtype=np.float32)
        images = np.reshape(images, (images.shape[0], 100, 100, 3), order='F')  # to RGB
        images -= wrong_mean
        features, _ = extractor.extract_from_images(images)
        pool4 = features[1][:, :, :, :]
        normed4 = pool4 / np.linalg.norm(pool4, axis=-1).reshape((images.shape[0], 1, 1, 1))
        real4 = normed4
        pool3 = features[0][:, 1:7, 1:7, :]
        normed3 = pool3 / np.linalg.norm(pool3, axis=-1).reshape((images.shape[0], 6, 6, 1))
        real3 = normed3

        real_dis4 = []
        for idx4 in range(176):
            vc4 = centers4[idx4, :]
            real_temp4 = np.linalg.norm(real4 - vc4, axis=-1)
            real_dis4.append(real_temp4)
        real_dis4 = np.stack(real_dis4, axis=-1)

        real_dis3 = []
        for idx3 in range(223):
            vc3 = centers3[idx3, :]
            real_temp3 = np.linalg.norm(np.abs(real3 - vc3), axis=-1)
            real_dis3.append(real_temp3)
        real_dis3 = np.stack(real_dis3, axis=-1)

        real_code4 = real_dis4 < thres
        real_code3 = real_dis3 < thres

        temp = json.dumps({'real_code4': real_code4.tolist(), 'real_code3': real_code3.tolist()})
        target = open('vc' + str(vc) + 'real', 'w')
        target.write(temp)
        target.close()

    # json.dump({'fake_dis4': fake_dis4.tolist(), 'fake_dis3': fake_dis3.tolist()}, open('cihang_vc' + str(vc) + 'fake_dis', 'w'))
    # json.dump({'real_dis4': real_dis4.tolist(), 'real_dis3': real_dis3.tolist()}, open('vc' + str(vc) + 'real_dis', 'w'))
    # json.dump({'fake_feat4': fake4.tolist(), 'fake_feat3': fake3.tolist()}, open('cihang_vc' + str(vc) + 'fake_feat', 'w'))
    # json.dump({'real_feat4': real4.tolist(), 'real_feat3': real3.tolist()}, open('vc' + str(vc) + 'real_feat', 'w'))