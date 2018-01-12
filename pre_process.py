# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os    
import nibabel
import numpy as np
from util.data_process import *
from util.parse_config import parse_config

def data_set_crop(read_folder, save_folder, modalities):
    """
    Crop the data with a bounding box that is automatically calculated based on the nonzero region
    inputs:
        read_folder: the folder containing the input data
        save_folder: the folder to save the cropped data
        modalities: modalities of the data, e.g., ['Flair', 'T1c', 'OT']
    """
    if(not os.path.isdir(save_folder)):
        os.mkdir(save_folder)
    patient_list = os.listdir(read_folder)
    patient_list = [x for x in patient_list if modalities[0] in x]
    patient_list = ['_'.join(x.split('_')[:-1]) for x in patient_list]
    margin = 5
    
    print('patient number ', len(patient_list))
    for patient_name in patient_list:
        print(patient_name)
        imgs = []
        for mod in modalities:
            img_name = os.path.join(read_folder, "{0:}_{1:}.nii.gz".format(patient_name, mod))
            img = load_nifty_volume_as_array(img_name)
            imgs.append(img)

        [d_idxes, h_idxes, w_idxes] = np.nonzero(imgs[0])
        mind = d_idxes.min() - margin; maxd = d_idxes.max() + margin
        minh = h_idxes.min() - margin; maxh = h_idxes.max() + margin
        minw = w_idxes.min() - margin; maxw = w_idxes.max() + margin
        for mod_idx in range(len(modalities)):
            roi_volume = imgs[mod_idx][np.ix_(range(mind, maxd), range(minh, maxh), range(minw, maxw))]
            save_name = "{0:}_{1:}.nii.gz".format(patient_name, modalities[mod_idx])
            save_name = os.path.join(save_folder, save_name)
            save_array_as_nifty_volume(roi_volume, save_name)

if __name__ == "__main__":
    data_read_root = "/home/guotai/data/brats_docker_data/origin"
    data_save_root = "/home/guotai/data/brats_docker_data/pre_process"
    modalities = ['Flair', 'T1c', 'T1', 'T2', 'OT']
    data_set_crop(data_read_root,data_save_root, modalities)

    
