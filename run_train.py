# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers
import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
from data_io.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
from net.MSNet import MSNet
from niftynet.network.unet import UNet3D
from niftynet.network.highres3dnet import HighRes3DNet
from niftynet.layer.loss_segmentation import LossFunction

import pickle
class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        if name == 'HighRes3DNet':
            return HighRes3DNet
        if name == 'UNet3D':
            return UNet3D
        print('unsupported network:', name)
        exit()
        

def run(stage, config_file):
    # construct graph
    config = parse_config(config_file)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
    config_test  = config['testing']  
     
    if(stage == 'train'):
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
   
    w_regularizer = None; b_regularizer = None
    if(stage == 'train'):
        w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
        b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes = class_num,
        w_regularizer = w_regularizer,
        b_regularizer = b_regularizer,
        name = net_name)

    if(net_type == 'MSNet'):
        net.set_params(config_net)
    
    loss_func = LossFunction(n_class=class_num)
    predicty = net(x, is_training = True)
    proby = tf.nn.softmax(predicty)
    loss = loss_func(predicty, y, weight_map = w)
    print('size of predicty:',predicty)
    
    # Initialize session and saver
    if(stage == 'train'):
        lr = config_train.get('learning_rate', 1e-3)
        opt_step = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())  
    saver = tf.train.Saver()
    
    loader = DataLoader()
    loader.set_params(config_data)
    loader.load_data()

    if(stage == 'train'):
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
                    tempp = train_pair['probs']
                    dice = loss.eval(feed_dict ={x:tempx, w:tempw, y:tempy})
                    batch_dice_list.append(dice)
                batch_dice = np.asarray(batch_dice_list, np.float32).mean()
                t = time.strftime('%X %x %Z')
                print(t, 'n', n,'loss', batch_dice)
                loss_list.append(batch_dice)
                np.savetxt(loss_file, np.asarray(loss_list))
            if((n+1)%config_train['snapshot_iteration']  == 0):
                saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(n+1))
    else:
        saver.restore(sess, config_test['model_file'])
        image_num = loader.get_total_image_number()
        test_slice_direction = config_test.get('test_slice_direction', 'all')
        save_folder = config_test['save_folder']
        test_time = []
        for i in range(image_num):
            [test_imgs, test_weight, temp_name] = loader.get_image_data_with_name(i)
            down_sample = config_test.get('down_sample_rate', 1.0)
            if(down_sample == 1.0):
                temp_imgs = test_imgs
                temp_weight = test_weight
            else:
                temp_imgs = []
                for mod in range(len(test_imgs)):
                    temp_imgs.append(ndimage.interpolation.zoom(test_imgs[mod],1.0/down_sample, order = 1))
                temp_weight = ndimage.interpolation.zoom(test_weight,1.0/down_sample, order = 1)
            t0 = time.time()
            if(net_type == 'HighRes3DNet' or net_type == 'UNet3D'):
                temp_prob = volume_probability_prediction_3d_roi(temp_imgs, data_shape, \
                    label_shape, data_channel, class_num, batch_size, sess, proby, x)
            else:
                temp_prob = test_one_image(temp_imgs, data_shape, label_shape, data_channel, class_num,
                    batch_size, test_slice_direction, sess, proby, x)
            temp_time = time.time() - t0
            test_time.append(temp_time)
            temp_label =  np.asarray(np.argmax(temp_prob, axis = 3), np.uint16)
            temp_label[temp_weight==0] = 0
            label_convert_source = config_test.get('label_convert_source', None)
            label_convert_target = config_test.get('label_convert_target', None)
            if(label_convert_source and label_convert_target):
                assert(len(label_convert_source) == len(label_convert_target))
                temp_label = convert_label(temp_label, label_convert_source, label_convert_target)
            if(down_sample != 1.0):
                temp_label = resize_3D_volume_to_given_shape(temp_label, test_weight.shape, order = 0)
            save_array_as_nifty_volume(temp_label, save_folder+"/{0:}.nii.gz".format(temp_name))
            # save probability
            save_prob = config_test.get('save_prob', False)
            if(save_prob):
                fg_prob = temp_prob[:,:,:,1]
                fg_prob = np.reshape(fg_prob, temp_label.shape)
                save_array_as_nifty_volume(fg_prob, save_folder+"_prob/{0:}.nii.gz".format(temp_name))
#             pickle.dump(temp_prob, open(save_folder+"_prob/{0:}.p".format(temp_name), 'w'))
        test_time = np.asarray(test_time)
        print('test time', test_time.mean(), test_time.std())
        np.savetxt(save_folder + '/test_time.txt', test_time)
    
    sess.close()
    
if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print('Number of arguments should be 3. e.g.')
        print('    python run_train.py train config.txt')
        exit()
    stage = str(sys.argv[1])
    config_file = str(sys.argv[2])
    assert(os.path.isfile(config_file))
    run(stage, config_file)
    
    
