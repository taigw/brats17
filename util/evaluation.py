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
import sys
sys.path.append('./')
import numpy as np
from util.data_process import load_3d_volume_as_array, binary_dice3d

def get_ground_truth_names(g_folder, patient_names_file, year = 15):
    assert(year==15 or year == 17)
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
    full_gt_names = []
    for patient_name in patient_names:
        patient_dir = os.path.join(g_folder, patient_name)
        img_names   = os.listdir(patient_dir)
        gt_name = None
        for img_name in img_names:
            if(year == 15):
                if 'OT.' in img_name:
                    gt_name = img_name + '/' + img_name + '.mha'
                    break
            else:
                if 'seg.' in img_name:
                    gt_name = img_name
                    break
        gt_name = os.path.join(patient_dir, gt_name)
        full_gt_names.append(gt_name)
    return full_gt_names

def get_segmentation_names(seg_folder, patient_names_file):
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
    full_seg_names = []
    for patient_name in patient_names:
        seg_name = os.path.join(seg_folder, patient_name + '.nii.gz')
        full_seg_names.append(seg_name)
    return full_seg_names

def dice_of_brats_data_set(gt_names, seg_names, type_idx):
    assert(len(gt_names) == len(seg_names))
    dice_all_data = []
    for i in range(len(gt_names)):
        g_volume = load_3d_volume_as_array(gt_names[i])
        s_volume = load_3d_volume_as_array(seg_names[i])
        dice_one_volume = []
        if(type_idx ==0): # whole tumor
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        elif(type_idx == 1): # tumor core
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        else:
            for label in [1, 2, 3, 4]: # dice of each class
                temp_dice = binary_dice3d(s_volume == label, g_volume == label)
                dice_one_volume.append(temp_dice)
        dice_all_data.append(dice_one_volume)
    return dice_all_data
    
if __name__ == '__main__':
    year = 15 # or 17
    
    if(year == 15):
        s_folder = 'result15'
        g_folder = '/home/guotwang/data/BRATS2015_Training'
        patient_names_file = 'config15/test_names.txt'
    else:
        s_folder = 'result17'
        g_folder = '/home/guotwang/data/Brats17TrainingData'
        patient_names_file = 'config15/test_names.txt'

    test_types = ['whole','core', 'all']
    gt_names  = get_ground_truth_names(g_folder, patient_names_file, year)
    seg_names = get_segmentation_names(s_folder, patient_names_file)
    for type_idx in range(3):
        dice = dice_of_brats_data_set(gt_names, seg_names, type_idx)
        dice = np.asarray(dice)
        dice_mean = dice.mean(axis = 0)
        dice_std  = dice.std(axis  = 0)
        test_type = test_types[type_idx]
        np.savetxt(s_folder + '/dice_{0:}.txt'.format(test_type), dice)
        np.savetxt(s_folder + '/dice_{0:}_mean.txt'.format(test_type), dice_mean)
        np.savetxt(s_folder + '/dice_{0:}_std.txt'.format(test_type), dice_std)
        print('tissue type', test_type)
        if(test_type == 'all'):
            print('tissue label', [1, 2, 3, 4])
        print('dice mean  ', dice_mean)
        print('dice std   ', dice_std)
 
