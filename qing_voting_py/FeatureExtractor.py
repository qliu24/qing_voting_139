import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.slim as slim
from datetime import datetime
import network as vgg
import os

class FeatureExtractor:
    def __init__(self, cache_folder, batch_size=1, which_layer='pool4', which_snapshot=200000, from_scratch=False):
        # params
        self.batch_size = batch_size
        self.scale_size = vgg.vgg_16.default_image_size
        self.img_mean = np.array([104., 117., 124.])

        # Runtime params
        checkpoints_dir = os.path.join(cache_folder, 'checkpoints')
        with tf.device('/cpu:0'):
            self.input_images = tf.placeholder(tf.float32, [self.batch_size, self.scale_size, self.scale_size, 3])

        with tf.variable_scope('vgg_16', reuse=False):
            with slim.arg_scope(vgg.vgg_arg_scope(bn=True)):
                _, vgg_end_points = vgg.vgg_16(self.input_images, is_training=True)
                
        self.features = vgg_end_points['vgg_16/' + which_layer]  # TODO

        # Create restorer and saver
        restorer = self.get_init_restorer(bn=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session(config=config)
        print(str(datetime.now()) + ': Start Init')
        if which_snapshot == 0:  # Start from a pre-trained vgg ckpt
            if from_scratch:
                self.sess.run(init_op)
            else:
                restorer.restore(self.sess, os.path.join(checkpoints_dir, 'vgg_16.ckpt'))
        else:  # Start from the last time
            restorer.restore(self.sess, os.path.join(checkpoints_dir, 'fine_tuned-' + str(which_snapshot)))
        print(str(datetime.now()) + ': Finish Init')
        
        
    def extract_feature_file(self, file_path):
        if isinstance(file_path, (list, tuple)):
            assert(self.batch_size == len(file_path))
        else:
            assert(isinstance(file_path, str))
            assert(self.batch_size == 1)
            file_path = [file_path]
            
        batch_images = np.ndarray([self.batch_size, self.scale_size, self.scale_size, 3])
        for ii in range(self.batch_size):
            img = cv2.imread(file_path[ii])
            h, w, c = img.shape
            assert c == 3
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (self.scale_size, self.scale_size))
            img = img.astype(np.float32)
            img -= self.img_mean
            batch_images[ii] = img
            
        print('feed network')
        feed_dict = {self.input_images: batch_images}
        feature_list = self.sess.run(self.features, feed_dict=feed_dict)
        return feature_list
    
    
    def get_init_restorer(self, bn=False, vc=False):
        """Returns a function run by the chief worker to warm-start the training."""
        checkpoint_exclude_scopes = ['vgg_16/fc8']
        if not bn:
            checkpoint_exclude_scopes.append('BatchNorm')  # restore bn params
        if not vc:
            checkpoint_exclude_scopes.append('vc_centers')  # restore bn params
        
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
        
        variables_to_restore = []
        # for var in slim.get_model_variables():
        for var in tf.global_variables():
            excluded = False
            for exclusion in exclusions:
                if exclusion in var.op.name:
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        # variables_to_restore += [var for var in tf.global_variables() if 'vc_centers' in var.op.name]  # always restore VC
        return tf.train.Saver(variables_to_restore)
    
    
    def extract_feature_image(self, img):
        assert(self.batch_size == 1)
        h, w, c = img.shape
        assert c == 3
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (self.scale_size, self.scale_size))
        img = img.astype(np.float32)
        img -= self.img_mean
            
        feed_dict = {self.input_images: [img]}
        out_features = self.sess.run(self.features, feed_dict=feed_dict)
        return out_features
    
    def close_sess(self):
        self.sess.close()