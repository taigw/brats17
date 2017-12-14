# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from util.data_process import load_nifty_volume_as_array, binary_dice3d

def dice_of_brats_data_set(s_folder, g_folder, patient_names_file, type_idx):
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content] 
    dice_all_data = []
    for i in range(len(patient_names)):
        s_name = os.path.join(s_folder, patient_names[i] + '.nii.gz')
        g_name = os.path.join(g_folder, patient_names[i] + '_seg.nii.gz')
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
    return dice_all_data
    
if __name__ == '__main__':
    s_folder = 'results'
    g_folder = 'data/Brats17TrainingData_crop_renamed'
    patient_names_file = 'config/test_names_example.txt'
    test_types = ['whole','core', 'all']
    for type_idx in range(3):
        dice = dice_of_brats_data_set(s_folder, g_folder, patient_names_file, type_idx)
        dice = np.asarray(dice)
        dice_mean = dice.mean(axis = 0)
        dice_std  = dice.std(axis  = 0)
        test_type = test_types[type_idx]
        np.savetxt(s_folder + '/dice_{0:}.txt'.format(test_type), dice)
        np.savetxt(s_folder + '/dice_{0:}_mean.txt'.format(test_type), dice_mean)
        np.savetxt(s_folder + '/dice_{0:}_std.txt'.format(test_type), dice_std)
        print('tissue type', test_type)
        if(test_type == 'all'):
            print('tissue label', [1, 2, 4])
        print('dice mean  ', dice_mean)
        print('dice std   ', dice_std)
 
