import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import sys
import os
import cv2
from tensorflow.python.client import timeline
from datetime import datetime
import network as vgg

def get_init_restorer(bn=False, vc=False):
    """Returns a function run by the chief worker to warm-start the training."""
    # checkpoint_exclude_scopes = ['vgg_16/fc8']
    checkpoint_exclude_scopes = []
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


class FeatureExtractor:
    def __init__(self, cache_folder, layer_names=['pool3/MaxPool:0', 'pool4/MaxPool:0']):
        # params
        self.batch_size = 20
        self.scale_size = vgg.vgg_16.default_image_size

        # Runtime params
        checkpoints_dir = os.path.join(cache_folder, 'checkpoints_vgg')
        tf.logging.set_verbosity(tf.logging.INFO)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with tf.device('/cpu:0'):
            self.input_images = tf.placeholder(tf.float32, [self.batch_size, self.scale_size, self.scale_size, 3])

        vgg_var_scope = 'vgg_16'
        with tf.variable_scope(vgg_var_scope, reuse=False):
            with slim.arg_scope(vgg.vgg_arg_scope(bn=False, is_training=False)):
                _, _ = vgg.vgg_16(self.input_images, is_training=False)
        # self.pool4 = vgg_end_points['vgg_16/pool4']
        # with tf.variable_scope('VC', reuse=False):
        #     self.tight_loss, self.tight_end_points = online_clustering(self.pool4, 512)
        # self.features = vgg_end_points['vgg_16/' + which_layer]  # TODO
        self.features = []
        for layer_name in layer_names:
            variable = tf.get_default_graph().get_tensor_by_name(vgg_var_scope + '/' + layer_name)
            self.features.append(variable)

        # Create restorer and saver
        restorer = get_init_restorer(bn=False, vc=False)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        init_op = tf.global_variables_initializer()
        # Run the session:
        self.sess = tf.Session(config=config)
        # init
        print(str(datetime.now()) + ': Start Init')
        
        restorer.restore(self.sess, os.path.join(checkpoints_dir, 'vgg_16.ckpt'))
        
        print(str(datetime.now()) + ': Finish Init')
        

    def extract_from_batch_images(self, batch_images):
        feed_dict = {self.input_images: batch_images}
        out_features = self.sess.run(self.features, feed_dict=feed_dict)
        return out_features

    def extract_from_images(self, images):
        feature_list = [[] for _ in range(len(self.features))]
        image_list = []
        for i in range(-(-images.shape[0] // self.batch_size)):
            batch_images = np.ndarray([self.batch_size, self.scale_size, self.scale_size, 3])
            for j in range(self.batch_size):
                # read paths
                if i * self.batch_size + j >= images.shape[0]:
                    break
                    
                batch_images[j] = images[i * self.batch_size + j, :, :, :]
                
            out_features = self.extract_from_batch_images(batch_images)
            for k in range(len(self.features)):
                feature_list[k].append(out_features[k])
                
            image_list.append(batch_images)
            
        for k in range(len(self.features)):
            feature_list[k] = np.concatenate(feature_list[k])
            feature_list[k] = feature_list[k][:images.shape[0]]
        # features = np.concatenate(feature_list)
        out_images = np.concatenate(image_list)
        out_images = out_images[:images.shape[0], :]
        return feature_list, out_images
    


