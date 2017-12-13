# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import numpy as np
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
from tensorflow.contrib.data import Iterator
from util.data_loader import *
from util.data_process import *
from util.train_test_func import *
from util.parse_config import parse_config
from train import NetFactory


def get_net_configs(config, net_level = 'network1'):
    config_net = config.get(net_level, None)
    if(config_net is not None):
        net_configs = [config_net]
    else:
        config_net_ax = config[net_level + 'ax']
        config_net_sg = config[net_level + 'sg']
        config_net_cr = config[net_level + 'cr']
        net_configs = [config_net_ax, config_net_sg, config_net_cr]
    return net_configs

def get_net_dict(config_net, batch_size):
    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    data_shape  = config_net['data_shape']
    label_shape = config_net['label_shape']
    class_num   = config_net['class_num']
    full_data_shape = batch_size + data_shape
    
    net_class = NetFactory.create(net_type)
    net       = net_class(num_classes = class_num,w_regularizer = None,
                          b_regularizer = None, name = net_name)
    net.set_params(config_net)
    x         = tf.placeholder(tf.float32, shape = full_data_shape)
    predicty  = net(x, is_training = True)
    proby     = tf.nn.softmax(predicty)
    net_dict  = {'net':net, 'x':x, 'proby':proby}
    return net_dict

class TestAgent(object):
    def __init__(self, config, level = 0):
        network_prefixes = ['network1', 'network1', 'network3']
        self.batch_size = config['batch_size']
        config_net = config.get(network_prefixes[level], None)
        if(config_net is not None):
            self.net_configs = [config_net]
        else:
            config_net_ax = config[network_prefixes[level] + 'ax']
            config_net_sg = config[network_prefixes[level] + 'sg']
            config_net_cr = config[network_prefixes[level] + 'cr']
            self.net_configs = [config_net_ax, config_net_sg, config_net_cr]
        self.__build_networks()

    def __build_one_network(self, config_net):
        net_type    = config_net['net_type']
        net_name    = config_net['net_name']
        data_shape  = config_net['data_shape']
        label_shape = config_net['label_shape']
        class_num   = config_net['class_num']
        full_data_shape = [self.batch_size] + data_shape
        
        net_class = NetFactory.create(net_type)
        net       = net_class(num_classes = class_num,w_regularizer = None,
                              b_regularizer = None, name = net_name)
        net.set_params(config_net)
        x         = tf.placeholder(tf.float32, shape = full_data_shape)
        predicty  = net(x, is_training = True)
        proby     = tf.nn.softmax(predicty)
        net_dict  = {'net':net, 'x':x, 'proby':proby}
        return net_dict

    def __build_networks(self):
        nets = []
        for net_config in self.net_configs:
            net_dict = self.__build_one_network(net_config)
            nets.append(net_dict)
        self.net_dicts = nets

    def __restore_vars_of_one_net(self, sess, all_vars, config_net):
        net_name   = config_net['net_name']
        model_file = config_net['model_file']
        net_vars = [x for x in all_vars if x.name[0:len(net_name) + 1]==net_name + '/']
        saver = tf.train.Saver(net_vars)
        saver.restore(sess, model_file)

    def restore_variables(self, sess, all_vars):
        for net_config in self.net_configs:
            self.__restore_vars_of_one_net(sess, all_vars, net_config)

def test_one_volume(agent, sess, imgs):
    data_shapes, label_shapes  = [], []
    nets, inputs, outputs      = [], [], []
    for i in range(len(agent.net_configs)):
        data_shapes.append( agent.net_configs[i]['data_shape' ][:-1])
        label_shapes.append(agent.net_configs[i]['label_shape'][:-1])
        nets.append(   agent.net_dicts[i]['net'  ])
        inputs.append( agent.net_dicts[i]['x'    ])
        outputs.append(agent.net_dicts[i]['proby'])
    if(len(data_shapes) == 1):
        data_shapes  = data_shapes * 3
        label_shapes = label_shapes * 3
        nets    = nets * 3
        inputs  = inputs * 3
        outputs = outputs * 3
    class_num = agent.net_configs[0]['class_num']
    chann_num = data_shapes[0][-1]
    prob = test_one_image_three_nets_adaptive_shape(imgs, data_shapes, label_shapes,
                chann_num , class_num, agent.batch_size, sess, nets, outputs, inputs, shape_mode = 1)
    return prob


def test(config_file):
    # 1, load configure file
    config = parse_config(config_file)
    config_data = config['data']
    config_test = config['testing']  
    batch_size  = config_test.get('batch_size', 5)
    config['batch_size'] = batch_size
    
    config_levels, net_levels  = [], []
    config_level1 = get_net_configs(config, 'network1')
    net_level1 = []
    for config in config_level1:
    
    config_net1 = config.get('network1', None)
    if(config_net1 is not None):
        config_net1 = [config_net1]
    else:
        config_net1_ax = config[network_prefixes[level] + 'ax']
        config_net1_sg = config[network_prefixes[level] + 'sg']
        config_net1_cr = config[network_prefixes[level] + 'cr']
        config_net1 = [config_net1_ax, config_net1_sg, config_net1_cr]
    test_wt = TestAgent(config, 0)
    if(config_test.get('whole_tumor_only', False) is False):
        test_tc = TestAgent(config, 1)
        test_en = TestAgent(config, 2)
        
    all_vars = tf.global_variables()
    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())
    test_wt.restore_variables(sess, all_vars)
    if(config_test.get('whole_tumor_only', False) is False):
        test_tc.restore_variables(sess, all_vars)
        test_en.restore_variables(sess, all_vars)

    # 3, load test images
    dataloader = DataLoader(config_data)
    dataloader.load_data()
    image_num = dataloader.get_total_image_number()
    
    # 4, start to test
    test_slice_direction = config_test.get('test_slice_direction', 'all')
    save_folder = config_test['save_folder']
    test_time = []
    struct = ndimage.generate_binary_structure(3, 2)
    margin = config_test.get('roi_patch_margin', 5)
    
    for i in range(image_num):
        [imgs, weight, temp_name] = dataloader.get_image_data_with_name(i)
        t0 = time.time()
        groi = get_roi(weight > 0, margin)
        temp_imgs = [x[np.ix_(range(groi[0], groi[1]), range(groi[2], groi[3]), range(groi[4], groi[5]))] \
                        for x in imgs]
        temp_weight = weight[np.ix_(range(groi[0], groi[1]), range(groi[2], groi[3]), range(groi[4], groi[5]))]

        poob1 = test_one_volume(test_wt, sess, temp_imgs)
        pred1 = np.asarray(np.argmax(prob1, axis = 3), np.uint16)
        pred1 = pred1 * temp_weight

        wt_threshold = 2000
        if(config_test.get('whole_tumor_only', False) is True):
            pred1_lc = ndimage.morphology.binary_closing(pred1, structure = struct)
            pred1_lc = get_largest_two_component(pred1_lc, True, wt_threshold)
            out_label = pred1_lc
        else:
            # 4.2, test of 2nd network
            if(pred1.sum() == 0):
                print('net1 output is null', temp_name)
                roi2 = get_roi(temp_imgs[0] > 0, margin)
            else:
                pred1_lc = ndimage.morphology.binary_closing(pred1, structure = struct)
                pred1_lc = get_largest_two_component(pred1_lc, True, wt_threshold)
                roi2 = get_roi(pred1_lc, margin)
            sub_imgs = [x[np.ix_(range(roi2[0], roi2[1]), range(roi2[2], roi2[3]), range(roi2[4], roi2[5]))] \
                          for x in temp_imgs]
            sub_weight = temp_weight[np.ix_(range(roi2[0], roi2[1]), range(roi2[2], roi2[3]), range(roi2[4], roi2[5]))]
            prob2 = test_one_volume(test_tc, sess, sub_imgs)
            pred2 = np.asarray(np.argmax(prob2, axis = 3), np.uint16)
            pred2 = pred2 * sub_weight
             
            # 4.3, test of 3rd network
            if(pred2.sum() == 0):
                [roid, roih, roiw] = sub_imgs[0].shape
                roi3 = [0, roid, 0, roih, 0, roiw]
                subsub_imgs = sub_imgs
                subsub_weight = sub_weight
            else:
                pred2_lc = ndimage.morphology.binary_closing(pred2, structure = struct)
                pred2_lc = get_largest_two_component(pred2_lc)
                roi3 = get_roi(pred2_lc, margin)
                subsub_imgs = [x[np.ix_(range(roi3[0], roi3[1]), range(roi3[2], roi3[3]), range(roi3[4], roi3[5]))] \
                          for x in sub_imgs]
                subsub_weight = sub_weight[np.ix_(range(roi3[0], roi3[1]), range(roi3[2], roi3[3]), range(roi3[4], roi3[5]))]
            prob3 = test_one_volume(test_en, sess, subsub_imgs)
            pred3 = np.asarray(np.argmax(prob3, axis = 3), np.uint16)
            pred3 = pred3 * subsub_weight
             
            # 4.4, fuse results at 3 levels
            # convert subsub_label to full size (non-enhanced)
            label3_roi = np.zeros_like(pred2)
            label3_roi[np.ix_(range(roi3[0], roi3[1]), range(roi3[2], roi3[3]), range(roi3[4], roi3[5]))] = pred3
            label3 = np.zeros_like(pred1)
            label3[np.ix_(range(roi2[0], roi2[1]), range(roi2[2], roi2[3]), range(roi2[4], roi2[5]))] = label3_roi

            label2 = np.zeros_like(pred1)
            label2[np.ix_(range(roi2[0], roi2[1]), range(roi2[2], roi2[3]), range(roi2[4], roi2[5]))] = pred2

            label1_mask = (pred1 + label2 + label3) > 0
            label1_mask = ndimage.morphology.binary_closing(label1_mask, structure = struct)
            label1_mask = get_largest_two_component(label1_mask, False, wt_threshold)
            label1 = pred1 * label1_mask
            
            label2_3_mask = (label2 + label3) > 0
            label2_3_mask = label2_3_mask * label1_mask
            label2_3_mask = ndimage.morphology.binary_closing(label2_3_mask, structure = struct)
            label2_3_mask = remove_external_core(label1, label2_3_mask)
            if(label2_3_mask.sum() > 0):
                label2_3_mask = get_largest_two_component(label2_3_mask)
            label1 = (label1 + label2_3_mask) > 0
            label2 = label2_3_mask
            label3 = label2 * label3
            vox_3  = label3.sum() 
            if(0 < vox_3 and vox_3 < 30):
                print('ignored voxel number ', vox_3)
                label3 = np.zeros_like(label2)

            out_label = label1 * 2 
            out_label[label2>0] = 1
            out_label[label3>0] = 3
            out_label = np.asarray(out_label, np.int16)

            # 4.5, convert label and save output
            label_convert_source = config_test.get('label_convert_source', None)
            label_convert_target = config_test.get('label_convert_target', None)
            if(label_convert_source and label_convert_target):
                assert(len(label_convert_source) == len(label_convert_target))
                out_label = convert_label(out_label, label_convert_source, label_convert_target)

        test_time.append(time.time() - t0)
        final_label = np.zeros_like(weight, np.int16)
        final_label[np.ix_(range(groi[0], groi[1]), range(groi[2], groi[3]), range(groi[4], groi[5]))] = out_label

        save_array_as_nifty_volume(final_label, save_folder+"/{0:}.nii.gz".format(temp_name))
        print(temp_name)
    test_time = np.asarray(test_time)
    print('test time', test_time.mean())
    np.savetxt(save_folder + '/test_time.txt', test_time)
    sess.close()
      
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python test.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    test(config_file)
    
    
