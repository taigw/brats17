# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
from tensorflow.contrib.data import Iterator
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction
from util.image_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
from util.MSNet import MSNet


import pickle
class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        print('unsupported network:', name)
        exit()

def train(config_file):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
     
    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    class_num   = config_net['class_num']
    batch_size  = config_data.get('batch_size', 5)
   
    # 2, construct graph
    full_data_shape  = [batch_size] + config_data['data_shape']
    full_label_shape = [batch_size] + config_data['label_shape']
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_label_shape)
    y = tf.placeholder(tf.int64,   shape = full_label_shape)
   
    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes = class_num,
                    w_regularizer = w_regularizer,
                    b_regularizer = b_regularizer,
                    name = net_name)
    net.set_params(config_net)
    predicty = net(x, is_training = True)
    proby    = tf.nn.softmax(predicty)
    
    loss_func = LossFunction(n_class=class_num)
    loss = loss_func(predicty, y, weight_map = w)
    print('size of predicty:',predicty)
    
    # 3, initialize session and saver
    lr = config_train.get('learning_rate', 1e-3)
    opt_step = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())  
    saver = tf.train.Saver()
    
    loader = DataLoader(config_data)
    train_data = loader.get_dataset('train', shuffle = True)
    batch_per_epoch = loader.get_batch_per_epoch()
    print('batch per epoch', batch_per_epoch)
    train_iterator = Iterator.from_structure(train_data.output_types,
                                             train_data.output_shapes)
    next_train_batch = train_iterator.get_next()
    train_init_op  = train_iterator.make_initializer(train_data)
    
    # 4, start to train
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_it  = config_train.get('start_iteration', 0)
    if( start_it > 0):
        saver.restore(sess, config_train['model_pre_trained'])
    loss_list, temp_loss_list = [], []
    for n in range(start_it, config_train['maximal_iteration']):
        if((n-start_it)%batch_per_epoch == 0):
            sess.run(train_init_op)
        one_batch = sess.run(next_train_batch)
        feed_dict = {x:one_batch['image'], w:one_batch['weight'], y:one_batch['label']}
        opt_step.run(session = sess, feed_dict=feed_dict)

        loss_value = loss.eval(feed_dict = feed_dict)
        temp_loss_list.append(loss_value)
        if((n+1)%config_train['loss_display_iteration'] == 0):
            avg_loss = np.asarray(temp_loss_list, np.float32).mean()
            t = time.strftime('%X %x %Z')
            print(t, 'iter', n,'loss', avg_loss)
            loss_list.append(avg_loss)
            np.savetxt(loss_file, np.asarray(loss_list))
            temp_loss_list = []
        if((n+1)%config_train['snapshot_iteration']  == 0):
            saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(n+1))
    sess.close()
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    train(config_file)
