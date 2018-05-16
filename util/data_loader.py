# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function

import os
import random
import nibabel
import numpy as np
from scipy import ndimage
from util.data_process import *

class DataLoader():
    def __init__(self, config):
        """
        Initialize the calss instance
        inputs:
            config: a dictionary representing parameters
        """
        self.config    = config
        self.data_root = config['data_root'] if type(config['data_root']) is list else [config['data_root']]
        self.modality_postfix     = config.get('modality_postfix', ['flair','t1', 't1ce', 't2'])
        self.intensity_normalize  = config.get('intensity_normalize', [True, True, True, True])
        self.with_ground_truth    = config.get('with_ground_truth', False)
        self.label_convert_source = config.get('label_convert_source', None)
        self.label_convert_target = config.get('label_convert_target', None)
        self.label_postfix = config.get('label_postfix', 'seg')
        self.file_postfix  = config.get('file_postfix', 'nii.gz')
        self.data_names    = config.get('data_names', None)
        self.data_num      = config.get('data_num', None)
        self.data_resize   = config.get('data_resize', None)
        self.with_flip     = config.get('with_flip', False)

        if(self.label_convert_source and self.label_convert_target):
            assert(len(self.label_convert_source) == len(self.label_convert_target))
            
    def __get_patient_names(self):
        """
        get the list of patient names, if self.data_names id not None, then load patient 
        names from that file, otherwise search all the names automatically in data_root
        """
        # use pre-defined patient names
        if(self.data_names is not None):
            assert(os.path.isfile(self.data_names))
            with open(self.data_names) as f:
                content = f.readlines()
            patient_names = [x.strip() for x in content]
        # use all the patient names in data_root
        else:
            patient_names = os.listdir(self.data_root[0])
            patient_names = [name for name in patient_names if 'brats' in name.lower()]
        return patient_names

    def __load_one_volume(self, patient_name, mod):
        patient_dir = os.path.join(self.data_root[0], patient_name)
        # for bats17
        if('nii' in self.file_postfix):
            image_names = os.listdir(patient_dir)
            volume_name = None
            for image_name in image_names:
                if(mod + '.' in image_name):
                    volume_name = image_name
                    break
        # for brats15
        else:
            img_file_dirs = os.listdir(patient_dir)
            volume_name  = None
            for img_file_dir in img_file_dirs:
                if(mod+'.' in img_file_dir):
                    volume_name = img_file_dir + '/' + img_file_dir + '.' + self.file_postfix
                    break
        assert(volume_name is not None)
        volume_name = os.path.join(patient_dir, volume_name)
        volume = load_3d_volume_as_array(volume_name)
        return volume, volume_name

    def load_data(self):
        """
        load all the training/testing data
        """
        self.patient_names = self.__get_patient_names()
        assert(len(self.patient_names)  > 0)
        ImageNames = []
        X = []
        W = []
        Y = []
        bbox  = []
        in_size = []
        data_num = self.data_num if (self.data_num is not None) else len(self.patient_names)
        for i in range(data_num):
            volume_list = []
            volume_name_list = []
            for mod_idx in range(len(self.modality_postfix)):
                volume, volume_name = self.__load_one_volume(self.patient_names[i], self.modality_postfix[mod_idx])
                if(mod_idx == 0):
                    margin = 5
                    bbmin, bbmax = get_ND_bounding_box(volume, margin)
                    volume_size  = volume.shape
                volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
                if(self.data_resize):
                    volume = resize_3D_volume_to_given_shape(volume, self.data_resize, 1)
                if(mod_idx ==0):
                    weight = np.asarray(volume > 0, np.float32)
                if(self.intensity_normalize[mod_idx]):
                    volume = itensity_normalize_one_volume(volume)
                volume_list.append(volume)
                volume_name_list.append(volume_name)
            ImageNames.append(volume_name_list)
            X.append(volume_list)
            W.append(weight)
            bbox.append([bbmin, bbmax])
            in_size.append(volume_size)
            if(self.with_ground_truth):
                label, _ = self.__load_one_volume(self.patient_names[i], self.label_postfix)
                label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax)
                if(self.data_resize):
                    label = resize_3D_volume_to_given_shape(label, self.data_resize, 0)
                Y.append(label)
            if((i+1)%50 == 0 or (i+1) == data_num):
                print('Data load, {0:}% finished'.format((i+1)*100.0/data_num))
        self.image_names = ImageNames
        self.data   = X
        self.weight = W
        self.label  = Y
        self.bbox   = bbox
        self.in_size= in_size
    
    def get_subimage_batch(self):
        """
        sample a batch of image patches for segmentation. Only used for training
        """
        flag = False
        while(flag == False):
            batch = self.__get_one_batch()
            labels = batch['labels']
            if(labels.sum() > 0):
                flag = True
        return batch
    
    def __get_one_batch(self):
        """
        get a batch from training data
        """
        batch_size = self.config['batch_size']
        data_shape = self.config['data_shape']
        label_shape = self.config['label_shape']
        down_sample_rate = self.config.get('down_sample_rate', 1.0)
        data_slice_number = data_shape[0]
        label_slice_number = label_shape[0]
        batch_sample_model   = self.config.get('batch_sample_model', ('full', 'valid', 'valid'))
        batch_slice_direction= self.config.get('batch_slice_direction', 'axial') # axial, sagittal, coronal or random
        train_with_roi_patch = self.config.get('train_with_roi_patch', False)
        keep_roi_outside = self.config.get('keep_roi_outside', False)
        if(train_with_roi_patch):
            label_roi_mask    = self.config['label_roi_mask']
            roi_patch_margin  = self.config['roi_patch_margin']

        # return batch size: [batch_size, slice_num, slice_h, slice_w, moda_chnl]
        data_batch = []
        weight_batch = []
        label_batch = []
        slice_direction = batch_slice_direction
        if(slice_direction == 'random'):
            directions = ['axial', 'sagittal', 'coronal']
            idx = random.randint(0,2)
            slice_direction = directions[idx]
        for i in range(batch_size):
            if(self.with_flip):
                flip = random.random() > 0.5
            else:
                flip = False
            self.patient_id = random.randint(0, len(self.data)-1)
            data_volumes = [x for x in self.data[self.patient_id]]
            weight_volumes = [self.weight[self.patient_id]]
            boundingbox = None
            if(self.with_ground_truth):
                label_volumes = [self.label[self.patient_id]]
                if(train_with_roi_patch):
                    mask_volume = np.zeros_like(label_volumes[0])
                    for mask_label in label_roi_mask:
                        mask_volume = mask_volume + (label_volumes[0] == mask_label)
                    [d_idxes, h_idxes, w_idxes] = np.nonzero(mask_volume)
                    [D, H, W] = label_volumes[0].shape
                    mind = max(d_idxes.min() - roi_patch_margin, 0)
                    maxd = min(d_idxes.max() + roi_patch_margin, D)
                    minh = max(h_idxes.min() - roi_patch_margin, 0)
                    maxh = min(h_idxes.max() + roi_patch_margin, H)
                    minw = max(w_idxes.min() - roi_patch_margin, 0)
                    maxw = min(w_idxes.max() + roi_patch_margin, W)
                    if(keep_roi_outside):
                        boundingbox = [mind, maxd, minh, maxh, minw, maxw]
                    else:
                        for idx in range(len(data_volumes)):
                            data_volumes[idx] = data_volumes[idx][np.ix_(range(mind, maxd), 
                                                                     range(minh, maxh), 
                                                                     range(minw, maxw))]
                        for idx in range(len(weight_volumes)):
                            weight_volumes[idx] = weight_volumes[idx][np.ix_(range(mind, maxd), 
                                                                     range(minh, maxh), 
                                                                     range(minw, maxw))]
                        for idx in range(len(label_volumes)):
                            label_volumes[idx] = label_volumes[idx][np.ix_(range(mind, maxd), 
                                                                     range(minh, maxh), 
                                                                     range(minw, maxw))]

                if(self.label_convert_source and self.label_convert_target):
                    label_volumes[0] = convert_label(label_volumes[0], self.label_convert_source, self.label_convert_target)
        
            transposed_volumes = transpose_volumes(data_volumes, slice_direction)
            volume_shape = transposed_volumes[0].shape
            sub_data_shape = [data_slice_number, data_shape[1], data_shape[2]]
            sub_label_shape =[label_slice_number, label_shape[1], label_shape[2]]
            center_point = get_random_roi_sampling_center(volume_shape, sub_label_shape, batch_sample_model, boundingbox)
            sub_data = []
            for moda in range(len(transposed_volumes)):
                sub_data_moda = extract_roi_from_volume(transposed_volumes[moda],center_point,sub_data_shape)
                if(flip):
                    sub_data_moda = np.flip(sub_data_moda, -1)
                if(down_sample_rate != 1.0):
                    sub_data_moda = ndimage.interpolation.zoom(sub_data_moda, 1.0/down_sample_rate, order = 1)   
                sub_data.append(sub_data_moda)
            sub_data = np.asarray(sub_data)
            data_batch.append(sub_data)
            transposed_weight = transpose_volumes(weight_volumes, slice_direction)
            sub_weight = extract_roi_from_volume(transposed_weight[0],
                                                  center_point,
                                                  sub_label_shape,
                                                  fill = 'zero')
            
            if(flip):
                sub_weight = np.flip(sub_weight, -1)
            if(down_sample_rate != 1.0):
                    sub_weight = ndimage.interpolation.zoom(sub_weight, 1.0/down_sample_rate, order = 1)   
            weight_batch.append([sub_weight])
            if(self.with_ground_truth):
                tranposed_label = transpose_volumes(label_volumes, slice_direction)
                sub_label = extract_roi_from_volume(tranposed_label[0],
                                                     center_point,
                                                     sub_label_shape,
                                                     fill = 'zero')
                if(flip):
                    sub_label = np.flip(sub_label, -1)
                if(down_sample_rate != 1.0):
                    sub_label = ndimage.interpolation.zoom(sub_label, 1.0/down_sample_rate, order = 0)  
                label_batch.append([sub_label])
                    
        data_batch = np.asarray(data_batch, np.float32)
        weight_batch = np.asarray(weight_batch, np.float32)
        label_batch = np.asarray(label_batch, np.int64)
        batch = {}
        batch['images']  = np.transpose(data_batch,   [0, 2, 3, 4, 1])
        batch['weights'] = np.transpose(weight_batch, [0, 2, 3, 4, 1])
        batch['labels']  = np.transpose(label_batch,  [0, 2, 3, 4, 1])
        
        return batch
    
    def get_total_image_number(self):
        """
        get the toal number of images
        """
        return len(self.data)
    
    def get_image_data_with_name(self, i):
        """
        Used for testing, get one image data and patient name
        """
        return [self.data[i], self.weight[i], self.patient_names[i], self.image_names[i], self.bbox[i], self.in_size[i]]
