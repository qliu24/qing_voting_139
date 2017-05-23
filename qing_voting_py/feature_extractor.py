from utils import *
from tensorflow.python.client import timeline
from datetime import datetime
import network as vgg
from global_variables import *
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import random
import json
import networkx as nx
from scipy.optimize import linear_sum_assignment


class FeatureExtractor:
    def __init__(self, which_layer='pool4', which_snapshot=200000, from_scratch=False):
        # params
        self.batch_size = 50
        self.scale_size = vgg.vgg_16.default_image_size
        self.img_mean = np.array([104., 117., 124.])

        # Runtime params
        checkpoints_dir = os.path.join(g_cache_folder, 'checkpoints')
        tf.logging.set_verbosity(tf.logging.INFO)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with tf.device('/cpu:0'):
            self.input_images = tf.placeholder(tf.float32, [self.batch_size, self.scale_size, self.scale_size, 3])

        with tf.variable_scope('vgg_16', reuse=False):
            with slim.arg_scope(vgg.vgg_arg_scope(bn=True)):
                _, vgg_end_points = vgg.vgg_16(self.input_images, is_training=True)
                
        self.features = vgg_end_points['vgg_16/' + which_layer]  # TODO

        # Create restorer and saver
        restorer = get_init_restorer(bn=True)
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

    def extract_from_paths(self, paths):
        feature_list = []
        image_list = []
        for i in range(-(-len(paths) // self.batch_size)):
            batch_images = np.ndarray([self.batch_size, self.scale_size, self.scale_size, 3])
            for j in range(self.batch_size):
                # read paths
                if i * self.batch_size + j >= len(paths):
                    break
                img = cv2.imread(paths[i * self.batch_size + j])
                h, w, c = img.shape
                assert c == 3
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, (self.scale_size, self.scale_size))
                img = img.astype(np.float32)
                img -= self.img_mean
                
                batch_images[j] = img
                # batch_images[j] = process_image(img, paths[i * self.batch_size + j], augment=0)
                
            out_features = self.extract_from_batch_images(batch_images)
            feature_list.append(out_features)
            image_list.append(batch_images)

        features = np.concatenate(feature_list)
        images = np.concatenate(image_list)
        return features[:len(paths), :], images[:len(paths), :]

    def extract_from_batch_images(self, batch_images):
        feed_dict = {self.input_images: batch_images}
        # [out_features, out_end_points, out_tight_loss] = self.sess.run([self.features, self.tight_end_points, self.tight_loss], feed_dict=feed_dict)
        out_features = self.sess.run(self.features, feed_dict=feed_dict)
        return out_features
