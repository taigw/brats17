# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from util.data_loader import *

def data_set_crop_rename(source_folder, save_folder, modalities, crop = True):
    patient_list = os.listdir(source_folder)
    patient_list = [x for x in patient_list if os.path.isdir(source_folder+'/'+x)]
    margin = 5
    print('patient number ', len(patient_list))
    for i in range(len(patient_list)):
        patient_name = patient_list[i]
        print(patient_name)
        imgs = []
        for mod in modalities:
            img_name = "{0:}_{1:}.nii.gz".format(patient_name, mod)
            img_name = os.path.join(source_folder, patient_name, img_name)
            img = load_nifty_volume_as_array(img_name)
            imgs.append(img)
    
        if(crop):
            [d_idxes, h_idxes, w_idxes] = np.nonzero(imgs[0])
            mind = d_idxes.min() - margin; maxd = d_idxes.max() + margin
            minh = h_idxes.min() - margin; maxh = h_idxes.max() + margin
            minw = w_idxes.min() - margin; maxw = w_idxes.max() + margin
        for mod_idx in range(len(modalities)):
            if(crop):
                roi_volume = imgs[mod_idx][np.ix_(range(mind, maxd), range(minh, maxh), range(minw, maxw))]
            else:
                roi_volume = imgs[mod_idx]
            save_name = "{0:}_{1:}.nii.gz".format(patient_name, modalities[mod_idx])
            save_name = os.path.join(save_folder, save_name)
            save_array_as_nifty_volume(roi_volume, save_name)

if __name__ == "__main__":
    # 1, crop and rename training data
    modalities    = ['flair', 't1ce', 't1', 't2', 'seg']
    source_folder = 'data/Brats17TrainingData/HGG/'
    target_folder = 'data/Brats17TrainingData_crop_renamed/HGG/'
    data_set_crop_rename(source_folder,target_folder, modalities, crop = True)

    source_folder = 'data/Brats17TrainingData/LGG/'
    target_folder = 'data/Brats17TrainingData_crop_renamed/LGG/'
    data_set_crop_rename(source_folder,target_folder, modalities, crop = True)

    # 2, rename validation data
    modalities    = ['flair', 't1ce', 't1', 't2']
    source_folder = 'data/Brats17ValidationData/'
    target_folder = 'data/Brats17ValidationData_renamed/'
    data_set_crop_rename(source_folder,target_folder, modalities, crop = False)
