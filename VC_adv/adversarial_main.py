import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from datetime import datetime
import network as vgg
import math
import pickle
from scipy.spatial.distance import cdist
import scipy.io as sio
from myresize import myresize
import numpy as np
import sys
import os
import cv2
from scipy.misc import logsumexp

cat_to_idx = dict()
cat_to_idx['car'] = 436
cat_to_idx['aeroplane'] = 404
cat_to_idx['bicycle'] = 444
cat_to_idx['bus'] = 874
cat_to_idx['motorbike'] = 665
cat_to_idx['train'] = 466

dataset_suffix = 'mergelist_rand'
Dataset = dict()
Dataset['img_dir'] = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Images/{0}_imagenet/'
Dataset['anno_dir'] = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Annotations/{0}_imagenet/'
Dataset['gt_dir'] = '/export/home/qliu24/qing_voting_139/qing_voting_py/intermediate/ground_truth_data/'
Dataset['train_list'] = os.path.join(Dataset['gt_dir'], '{0}_'+ '{0}_train.txt'.format(dataset_suffix))
Dataset['test_list'] = os.path.join(Dataset['gt_dir'], '{0}_'+ '{0}_test.txt'.format(dataset_suffix))

######### config #############
scale_size = 224
cache_folder = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'

featDim = 512
img_mean = np.array([104., 117., 124.]).reshape(1,1,3)

category = 'car'
target = 'aeroplane'
target_idx = cat_to_idx[target]

img_dir = Dataset['img_dir'].format(category)
file_list = Dataset['test_list'].format(category)

save_dir = '/export/home/qliu24/VC_adv_data/qing/VGG_adv/car/'

##################### init VGG
def get_init_restorer():
    variables_to_restore = []
    # for var in slim.get_model_variables():
    for var in tf.global_variables():
        variables_to_restore.append(var)
    
    return tf.train.Saver(variables_to_restore)


checkpoints_dir = os.path.join(cache_folder, 'checkpoints_vgg')
tf.logging.set_verbosity(tf.logging.INFO)
with tf.device('/cpu:0'):
    input_images = tf.placeholder(tf.float32, [1, None, None, 3])

vgg_var_scope = 'vgg_16'
with tf.variable_scope(vgg_var_scope, reuse=False):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, end_points = vgg.vgg_16(input_images)
        
restorer = get_init_restorer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
init_op = tf.global_variables_initializer()
sess = tf.Session(config=config)
print(str(datetime.now()) + ': Start Init')
restorer.restore(sess, os.path.join(checkpoints_dir, 'vgg_16.ckpt'))
print(str(datetime.now()) + ': Finish Init')

grad_ts_ls = []
for ii in range(end_points['vgg_16/fc8/reduced'].get_shape().as_list()[1]):
    grad_ts_ls.append(tf.gradients(end_points['vgg_16/fc8/reduced'][0,ii], input_images)[0])
    
target_grad_ts = grad_ts_ls[target_idx]

##################### load images
with open(file_list, 'r') as fh:
    content = fh.readlines()
    
img_list = [x.strip().split() for x in content]
img_num = len(img_list)
print('total number of images for {}: {}'.format(category, img_num))

for nn in range(519,img_num):
    print('Fooling image {0}:'.format(nn))
    file_img = os.path.join(img_dir, '{0}.JPEG'.format(img_list[nn][0]))
    assert(os.path.isfile(file_img))
    im = cv2.imread(file_img)
    im = im.astype(np.float32)
    im = myresize(im, scale_size, 'short')

    im_ori = np.copy(im)
    im -= img_mean
    im = im.reshape(np.concatenate([[1],im.shape]))

    r = np.zeros_like(im)
    itr = 0
    out = sess.run(end_points['vgg_16/fc8/reduced'], feed_dict={input_images: im})[0]
    out_prob = np.exp(out-logsumexp(out))
    pred_score = np.max(out)
    pred_idx = np.argmax(out)
    target_score = out[target_idx]
    target_prob = out_prob[target_idx]
    
    while target_prob < 0.9 and itr < 500:
        itr += 1
        pred_grad_ts = grad_ts_ls[pred_idx]

        grad = sess.run(target_grad_ts-pred_grad_ts, feed_dict={input_images: im+r})

        dr = (np.absolute(target_score-pred_score)+1)*grad/(np.linalg.norm(grad.ravel())**2)
        r += dr

        r_max = np.max(r)
        out = sess.run(end_points['vgg_16/fc8/reduced'], feed_dict={input_images: im+r})[0]
        out_prob = np.exp(out-logsumexp(out))

        sort_idx = np.argsort(-out)
        if sort_idx[0] == target_idx:
            pred_idx = sort_idx[1]
        else:
            pred_idx = sort_idx[0]

        pred_score = out[pred_idx]

        target_score = out[target_idx]
        target_prob = out_prob[target_idx]
        print('iteration {0}: max perturbation is {1:.2f}, target prob is {2:.2f}'.format(itr, r_max, target_prob))
        
    
    im_fool = im+r
    im_fool = np.squeeze(im_fool)
    im_fool += img_mean
    im_fool = np.clip(im_fool, 0, 255)
    
    fooling_image_name = os.path.join(save_dir, '{}.pickle'.format(img_list[nn][0]))
    with open(fooling_image_name, 'wb') as fh:
        pickle.dump([im_ori, r, im_fool], fh)
    