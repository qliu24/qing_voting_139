import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.slim as slim
from datetime import datetime
import network as vgg
import os

def get_init_restorer(bn=False, vc=False):
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = []

    if not bn:
        checkpoint_exclude_scopes.append('BatchNorm')  # restore bn params
    if not vc:
        checkpoint_exclude_scopes.append('vc_centers')  # restore bn params

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in tf.global_variables():
        excluded = False
        for exclusion in exclusions:
            if exclusion in var.op.name:
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
            
    return tf.train.Saver(variables_to_restore)

class FeatureExtractor:
    # keep the aspect ratio of the original image requires process one image at a time
    def __init__(self, cache_folder, which_net, which_layer, which_snapshot=0, from_scratch=False):
        # params
        self.batch_size = 1
        self.scale_size = vgg.vgg_16.default_image_size
        self.img_mean = np.array([104., 117., 124.])

        # Runtime params
        self.net_type = which_net
        with tf.device('/cpu:0'):
            self.input_images = tf.placeholder(tf.float32, [self.batch_size, None, None, 3])
        
        if which_net=='vgg16':
            checkpoints_dir = os.path.join(cache_folder, 'checkpoints_vgg')
            vgg_var_scope = 'vgg_16'
            
            if which_snapshot == 0:  # Start from a pre-trained vgg ckpt
                with tf.variable_scope(vgg_var_scope, reuse=False):
                    with slim.arg_scope(vgg.vgg_arg_scope(padding='SAME', bn=False, is_training=False)):
                        _, _ = vgg.vgg_16_part(self.input_images, is_training=False)
                    
                self.features = tf.get_default_graph().get_tensor_by_name(vgg_var_scope + '/' + which_layer + '/MaxPool:0')
                
                restorer = get_init_restorer(bn=False, vc=False)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                init_op = tf.global_variables_initializer()
                self.sess = tf.Session(config=config)
                
                print(str(datetime.now()) + ': Start Init')
                if from_scratch:
                    self.sess.run(init_op)
                else:
                    restorer.restore(self.sess, os.path.join(checkpoints_dir, 'vgg_16.ckpt'))
                    
            else:
                with tf.variable_scope(vgg_var_scope, reuse=False):
                    with slim.arg_scope(vgg.vgg_arg_scope(padding='SAME', bn=True, is_training=False)):
                        _, _ = vgg.vgg_16(self.input_images, is_training=False)
                        
                self.features = tf.get_default_graph().get_tensor_by_name(vgg_var_scope + '/' + which_layer + '/MaxPool:0')
                
                restorer = get_init_restorer(bn=True, vc=False)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                init_op = tf.global_variables_initializer()
                self.sess = tf.Session(config=config)
                
                print(str(datetime.now()) + ': Start Init')
                restorer.restore(self.sess, os.path.join(checkpoints_dir, 'fine_tuned-' + str(which_snapshot)))
            
        elif which_net=='alexnet':
            checkpoints_dir = os.path.join(cache_folder, 'checkpoints_alex')
            vgg_var_scope = 'vgg_16'
            with tf.variable_scope(vgg_var_scope, reuse=False):
                with slim.arg_scope(vgg.vgg_arg_scope(bn=True, is_training=False)):
                    _, _ = vgg.alexnet(self.input_images, is_training=False)
                    
            self.features = tf.get_default_graph().get_tensor_by_name(vgg_var_scope + '/' + which_layer + '/Relu:0')
            
            restorer = get_init_restorer(bn=True, vc=False)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            init_op = tf.global_variables_initializer()
            self.sess = tf.Session(config=config)
            
            print(str(datetime.now()) + ': Start Init')
            restorer.restore(self.sess, os.path.join(checkpoints_dir, 'fine_tuned-' + str(which_snapshot)))
            
        else:
            print('error: unknown net')
            return 0
        
        print(str(datetime.now()) + ': Finish Init')
    
    
    def extract_feature_image(self, img, is_gray=False, centered = False):
        assert(self.batch_size == 1)
        h, w, c = img.shape
        assert c == 3
        if is_gray:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            
        # img = cv2.resize(img, (self.scale_size, self.scale_size))
        
        img = img.astype(np.float32)
        if not centered:
            img -= self.img_mean
            
        feed_dict = {self.input_images: [img]}
        out_features = self.sess.run(self.features, feed_dict=feed_dict)
        return out_features
    
    