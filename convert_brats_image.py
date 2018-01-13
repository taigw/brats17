# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os    
import numpy as np
import SimpleITK as sitk
from util.parse_config import parse_config

def convert_brats15_data(input_folder, output_folder, modalities):
    """
    Copy the brats dataset to a single folder, and convert .mha files into .nii.gz files
    inputs:
        input_folder: the root folder of brats data, e.g. */BRATS2015_Training
        output_folder: the folder where the converted files will be writen
        modalities: modalities of files, e.g., ['Flair', 'T1c', 'T1', 'OT']
    """
    sub_folders = ['HGG', 'LGG']
    if(not os.path.isdir(output_folder)):
        os.mkdir(output_folder)
    for sub_folder in sub_folders:
        sub_dir = input_folder + '/' + sub_folder
        patients = os.listdir(sub_dir)
        for patient in patients:
            if 'brats' in patient:
                patient_dir = sub_dir + '/' + patient
                scan_dirs = os.listdir(patient_dir)
                for mod in modalities:
                    for scan_dir in scan_dirs:
                        if mod + '.' in scan_dir:
                            input_name = patient_dir + '/' + scan_dir + '/' + scan_dir + ".mha"
                            img = sitk.ReadImage(input_name)
                            save_name = "{0:}/{1:}_{2:}_{3:}.nii.gz".format(output_folder, sub_folder, patient, mod)
                            sitk.WriteImage(img, save_name)

if __name__ == "__main__":
    brats15_data_root = "/mnt/shared/guotai/data/BRATS2015_Training"
    data_save_root    = "/home/guotai/data/brats_docker_data/origin"
    modalities = ['Flair', 'T1c', 'T1', 'T2', 'OT']
    convert_brats15_data(brats15_data_root, data_save_root, modalities)
