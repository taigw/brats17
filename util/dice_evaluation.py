# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
sys.path.append('data_io')
from data_loader import load_nifty_volume_as_array
import numpy as np
from scipy import ndimage

def get_largest_component(img):
    s = ndimage.generate_binary_structure(3,1) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1)) 
    max_label = np.where(sizes == sizes.max())[0] + 1
    return labeled_array == max_label

def binary_dice3d(s,g):
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = 2.0*s0/(s1 + s2 + 0.00001)
    return dice

def dice_of_binary_volumes(s_name, g_name):
    s = load_nifty_volume_as_array(s_name)
    g = load_nifty_volume_as_array(g_name)
    dice = binary_dice3d(s, g)
    return dice

def dice_of_brats_data_set(s_folder, g_folder, patient_names_file, type_idx):
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content] 
    dice_all_data = []
    for i in range(len(patient_names)):
        s_name = os.path.join(s_folder, patient_names[i] + '.nii.gz')
        g_name = os.path.join(g_folder, patient_names[i] + '_Label.nii.gz')
        s_volume = load_nifty_volume_as_array(s_name)
        g_volume = load_nifty_volume_as_array(g_name)
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
            for label in [1, 2, 4]: # dice of each class
                temp_dice = binary_dice3d(s_volume == label, g_volume == label)
                dice_one_volume.append(temp_dice)
        dice_all_data.append(dice_one_volume)
        print(dice_one_volume[0])
    return dice_all_data
    
if __name__ == '__main__':
    s_folder = 'results/valid0_1'
    g_folder = '/home/wenqili/BRATS/test'
    patient_names_file = 'config/test_names.txt'
#s_folder = '/Users/guotaiwang/Documents/workspace/tf_project/NiftyNet/guotai_brats/deepigeos/result_crf_post'
#    g_folder = '/Users/guotaiwang/Documents/data/BRATS/BRATS2015_Train_croprename'
#    patient_names_file = '/Users/guotaiwang/Documents/data/BRATS/BRATS2015_Train_croprename/test_names2.txt'
    #patient_names_file = '/Users/guotaiwang/Documents/workspace/tf_project/NiftyNet/guotai_brats/deepigeos/temp_names.txt'
    test_types = ['whole','core', 'all']
    type_idx = 0
    
    dice = dice_of_brats_data_set(s_folder, g_folder, patient_names_file, type_idx)
    dice = np.asarray(dice)
    print(dice.shape)
    dice_mean = dice.mean(axis = 0)
    dice_std  = dice.std(axis = 0)
    test_type = test_types[type_idx]
    np.savetxt(s_folder + '/dice_{0:}.txt'.format(test_type), dice)
    np.savetxt(s_folder + '/dice_{0:}_mean.txt'.format(test_type), dice_mean)
    np.savetxt(s_folder + '/dice_{0:}_std.txt'.format(test_type), dice_std)
    print('dice mean ', dice_mean)
    print('dice std  ', dice_std)
 
