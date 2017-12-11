# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction
from util.data_loader import *
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
    # load configuration parameters
    config = parse_config(config_file)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
    config_test  = config['testing']  
     
    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    data_shape  = config_net['data_shape']
    label_shape = config_net['label_shape']
    data_channel= config_net['data_channel']
    class_num   = config_net['class_num']
    batch_size  = config_data.get('batch_size', 5)
   
    # construct graph
    print('data_channel',data_channel)
    full_data_shape = [batch_size] + data_shape + [data_channel]
    full_label_shape = [batch_size] + label_shape + [1]
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_label_shape)
    y = tf.placeholder(tf.int64, shape = full_label_shape)  
   
    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes = class_num,
        w_regularizer = w_regularizer,
        b_regularizer = b_regularizer,
        name = net_name)
    net.set_params(config_net)
    predicty = net(x, is_training = True)
    proby = tf.nn.softmax(predicty)
    
    loss_func = LossFunction(n_class=class_num)
    loss = loss_func(predicty, y, weight_map = w)
    print('size of predicty:',predicty)
    
    # Initialize session and saver
    lr = config_train.get('learning_rate', 1e-3)
    opt_step = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())  
    saver = tf.train.Saver()
    
    loader = DataLoader()
    loader.set_params(config_data)
    loader.load_data()

    loss_list = []
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_it = config_train.get('start_iteration', 0)
    if( start_it> 0):
        saver.restore(sess, config_train['model_pre_trained'])
    for n in range(start_it, config_train['maximal_iteration']):
        train_pair = loader.get_subimage_batch()
        tempx = train_pair['images']
        tempw = train_pair['weights']
        tempy = train_pair['labels']
        opt_step.run(session = sess, feed_dict={x:tempx, w: tempw, y:tempy})
          
        if(n%config_train['test_iteration'] == 0):
            batch_dice_list = []
            for step in range(config_train['test_step']):
                train_pair = loader.get_subimage_batch()
                tempx = train_pair['images']
                tempw = train_pair['weights']
                tempy = train_pair['labels']
                dice = loss.eval(feed_dict ={x:tempx, w:tempw, y:tempy})
                batch_dice_list.append(dice)
            batch_dice = np.asarray(batch_dice_list, np.float32).mean()
            t = time.strftime('%X %x %Z')
            print(t, 'n', n,'loss', batch_dice)
            loss_list.append(batch_dice)
            np.savetxt(loss_file, np.asarray(loss_list))
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
    
    
