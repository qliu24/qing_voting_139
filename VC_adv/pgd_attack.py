import network as vgg
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from datetime import datetime
import os

def get_init_restorer(bn=False, vc=False):
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ['vgg_16/fc8']

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

class LinfPGDAttack:
    def __init__(self, cache_folder, target, which_layer='pool4/MaxPool:0', epsilon=5, k=1000, a=0.01):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        
        self.batch_size = 50
        self.scale_size = 100
        self.epsilon = epsilon  # noise radius
        self.k = k  # iter number
        self.a = a  # learning rate
        
        
        with tf.device('/cpu:0'):
            self.x_input = tf.placeholder(tf.float32, [self.batch_size, self.scale_size, self.scale_size, 3])
            
        checkpoints_dir = os.path.join(cache_folder, 'checkpoints_vgg')
        vgg_var_scope = 'vgg_16'
        
        with tf.variable_scope(vgg_var_scope, reuse=False):
            with slim.arg_scope(vgg.vgg_arg_scope(padding='VALID', bn=False, is_training=False)):
                _, _ = vgg.vgg_16(self.x_input, is_training=False)

        restorer = get_init_restorer(bn=False, vc=False)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        print(str(datetime.now()) + ': Start Init')
        restorer.restore(self.sess, os.path.join(checkpoints_dir, 'vgg_16.ckpt'))
        print(str(datetime.now()) + ': Finish Init')
        
        features = tf.get_default_graph().get_tensor_by_name(vgg_var_scope + '/' + which_layer)
        features_norm = tf.reshape(features/tf.norm(features, axis=-1, keep_dims=True),[self.batch_size,-1])
        
        target = tf.reshape(tf.convert_to_tensor(target, dtype=tf.float32), [-1,1])
        self.loss = tf.reduce_sum(1-tf.matmul(features_norm, target))
        
        self.grad = tf.gradients(self.loss, self.x_input)[0]
        
        
    def perturb_batch_images(self, batch_images):
        x = np.copy(batch_images)
        for i in range(self.k):
            feed_dict = {self.x_input: x}
            grad = self.sess.run(self.grad, feed_dict=feed_dict)
            
            x -= self.a * np.sign(grad)
            x = np.clip(x, batch_images - self.epsilon, batch_images + self.epsilon) 
            # x = np.clip(x, 0, 255) # ensure valid pixel range
            if i%100==0:
                loss_i = self.sess.run(self.loss, feed_dict={self.x_input: x})
                print('iter {}: loss {}'.format(i, loss_i))

        return x

    def perturb(self, images):
        output_list = []
        for i in range(-(-images.shape[0] // self.batch_size)):
            batch_images = np.ndarray([self.batch_size, self.scale_size, self.scale_size, 3])
            for j in range(self.batch_size):
                # read paths
                if i * self.batch_size + j >= images.shape[0]:
                    break
                    
                batch_images[j] = images[i * self.batch_size + j, :, :, :]
                
            output_images = self.perturb_batch_images(batch_images)
            output_list.append(output_images)
            
        output_list = np.concatenate(output_list)
        output_list = output_list[0:images.shape[0], :]
        return output_list