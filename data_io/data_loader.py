# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import numpy as np
import nibabel
from PIL import Image
import random
from scipy import ndimage

def search_file_in_folder_list(folder_list, file_name):
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if(os.path.isfile(full_file_name)):
            file_exist = True
            break
    if(file_exist == False):
        raise ValueError('file not exist: {0:}'.format(file_name))
    return full_file_name

def load_nifty_volume_as_array(filename):
    # input shape [W, H, D]
    # output shape [D, H, W]
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    return data

def save_array_as_nifty_volume(data, filename):
    # numpy data shape [D, H, W]
    # nifty image shape [W, H, W]
    data = np.transpose(data, [2, 1, 0])
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, filename)

def itensity_normalize_one_volume(volume):
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size = volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def convert_label(in_volume, label_convert_source, label_convert_target):
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if(source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume>0] = convert_volume[mask_volume>0]
    return out_volume
        
def get_random_roi_sampling_center(input_shape, output_shape, sample_mode, bounding_box = None):
    center = []
    for i in range(len(input_shape)):
        if(sample_mode[i] == 'full'):
            if(bounding_box):
                x0 = bounding_box[i*2]; x1 = bounding_box[i*2 + 1]
            else:
                x0 = 0; x1 = input_shape[i]
        else:
            if(bounding_box):
                x0 = bounding_box[i*2] + int(output_shape[i]/2)   
                x1 = bounding_box[i*2+1] - int(output_shape[i]/2)   
            else:
                x0 = int(output_shape[i]/2)   
                x1 = input_shape[i] - x0
        if(x1 <= x0):
            centeri = int((x0 + x1)/2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center    

def transpose_volumes(volumes, slice_direction):
    if (slice_direction == 'axial'):
        tr_volumes = volumes
    elif(slice_direction == 'sagittal'):
        tr_volumes = [np.transpose(x, (2, 0, 1)) for x in volumes]
    elif(slice_direction == 'coronal'):
        tr_volumes = [np.transpose(x, (1, 0, 2)) for x in volumes]
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volumes
    return tr_volumes

def resize_3D_volume_to_given_shape(volume, out_shape, order = 3):  
    shape0=volume.shape
    scale_d=(out_shape[0]+0.0)/shape0[0]
    scale_h=(out_shape[1]+0.0)/shape0[1]
    scale_w=(out_shape[2]+0.0)/shape0[2]
    return ndimage.interpolation.zoom(volume,[scale_d,scale_h,scale_w], order = order)   
                                                    
def resize_ND_volume_to_given_shape(volume, out_shape, order = 3):  
    shape0=volume.shape
    assert(len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    return ndimage.interpolation.zoom(volume, scale, order = order) 

def extract_roi_from_volume(volume, in_center, output_shape, fill = 'random'):
    input_shape = volume.shape   
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x/2) for x in output_shape]
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output

def interaction_distance_processing(volume_list):
    # [img, seg, fg_dis, bg_dis]
    assert(len(volume_list) == 4)
    rand_img = np.random.normal(0.0, 1.0, size = volume_list[0].shape)
    img = volume_list[0]
    img[volume_list[2] < 1] = rand_img[volume_list[2] < 1] 
    img[volume_list[3] < 1] = rand_img[volume_list[3] < 1] 
    output_list  = [img, volume_list[1], volume_list[2], volume_list[3]]
    return output_list

class DataLoader():
    def __init__(self):
        pass
        
    def set_params(self, config):
        self.config = config
        self.data_root = config['data_root']
        self.modality_postfix = config['modality_postfix']
        self.intensity_normalize = config.get('intensity_normalize', None)
        self.label_postfix =  config.get('label_postfix', None)
        self.file_postfix = config['file_post_fix']
        self.data_names = config['data_names']
        self.data_num = config.get('data_num', 0)
        self.data_resize = config.get('data_resize', None)
        self.with_probability = config.get('with_probability', False)
        self.use_distance_mask = config.get('use_distance_mask', False)
        self.prob_postfix     = config.get('prob_postfix', None)
        self.with_ground_truth  = config.get('with_ground_truth', False)
        self.with_flip = config.get('with_flip', False)
        self.label_convert_source = self.config.get('label_convert_source', None)
        self.label_convert_target = self.config.get('label_convert_target', None)
        if(self.label_convert_source and self.label_convert_target):
            assert(len(self.label_convert_source) == len(self.label_convert_target))
        if(self.intensity_normalize == None):
            self.intensity_normalize = [True] * len(self.modality_postfix)
            
    def __get_patient_names(self):
        if(self.data_names):
            assert(os.path.isfile(self.data_names))
            with open(self.data_names) as f:
                content = f.readlines()
            patient_names = [x.strip() for x in content] 
        else: # load all image in data_root
            sub_dirs = [x[0] for x in os.walk(self.data_root[0])]
            print(sub_dirs)
            patient_names = []
            for sub_dir in sub_dirs:
                names = os.listdir(sub_dir)
                if(sub_dir == self.data_root[0]):
                    sub_patient_names = []
                    for x in names:
                        if(self.file_postfix in x):
                            idx = x.rfind('_')
                            xsplit = x[:idx]
                            sub_patient_names.append(xsplit)
                else:
                    sub_dir_name = sub_dir[len(self.data_root[0])+1:]
                    sub_patient_names = []
                    for x in names:
                        if(self.file_postfix in x):
                            idx = x.rfind('_')
                            xsplit = os.path.join(sub_dir_name,x[:idx])
                            sub_patient_names.append(xsplit)                    
                sub_patient_names = list(set(sub_patient_names))
                sub_patient_names.sort()
                patient_names.extend(sub_patient_names)   
        return patient_names
    
    def load_data(self):
        self.patient_names = self.__get_patient_names()
        X = []
        W = []
        Y = []
        P = []
        data_num = self.data_num if (self.data_num) else len(self.patient_names)
        for i in range(data_num):
            print(self.patient_names[i])
            volume_list = []
            for mod_idx in range(len(self.modality_postfix)):
                volume_name_short = self.patient_names[i] + '_' + self.modality_postfix[mod_idx] + '.' + self.file_postfix
                volume_name = search_file_in_folder_list(self.data_root, volume_name_short)
                volume = load_nifty_volume_as_array(volume_name)
                if(self.data_resize):
                    volume = resize_3D_volume_to_given_shape(volume, self.data_resize, 1)
                if(mod_idx == 0):
                    weight = np.asarray(volume > 0, np.float32)
                if(self.intensity_normalize[mod_idx]):
                    volume = itensity_normalize_one_volume(volume)
                volume_list.append(volume)
            if(self.use_distance_mask):
                volume_list = interaction_distance_processing(volume_list)
            X.append(volume_list)
            W.append(weight)
            if(self.with_ground_truth):
                label_name_short = self.patient_names[i] + '_' + self.label_postfix + '.' + self.file_postfix
                label_name = search_file_in_folder_list(self.data_root, label_name_short)
                label = load_nifty_volume_as_array(label_name)
                if(self.data_resize):
                    label = resize_3D_volume_to_given_shape(label, self.data_resize, 0)
                Y.append(label)
            if(self.with_probability):
                prob_name_short = self.patient_names[i] + '_' + self.prob_postfix + '.' + self.file_postfix
                prob_name = search_file_in_folder_list(self.data_root, prob_name_short)
                prob = load_nifty_volume_as_array(prob_name)
                if(self.data_resize):
                    prob = resize_3D_volume_to_given_shape(prob, self.data_resize, 1)
                P.append(prob)
                
        print('{0:} volumes have been loaded'.format(len(X)))
        self.data   = X
        self.weight = W
        self.label  = Y
        self.prob   = P
    
    
    def get_subimage_batch(self):
        '''
        sample a batch of image patches for segmentation. Only used for training
        '''
        flag = False
        while(flag == False):
            batch = self.__get_one_batch()
            labels = batch['labels']
            if(labels.sum() > 0):
                flag = True
        return batch
    
    def __get_one_batch(self):
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
            label_roi_mask = self.config['label_roi_mask']
            roi_patch_margin  = self.config['roi_patch_margin']

        # return batch size: [batch_size, slice_num, slice_h, slice_w, moda_chnl]
        data_batch = []
        weight_batch = []
        prob_batch = []
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
            if(self.with_probability):
                prob_volumes = [self.prob[self.patient_id]]
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
                        if(self.with_probability):
                            for idx in range(len(prob_volumes)):
                                prob_volumes[idx] = prob_volumes[idx][np.ix_(range(mind, maxd), 
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
            if(self.with_probability):
                tranposed_prob = transpose_volumes(prob_volumes, slice_direction)
                sub_prob = extract_roi_from_volume(tranposed_prob[0],
                                                     center_point,
                                                     sub_data_shape,
                                                     fill = 'zero')
                if(flip):
                    sub_prob = np.flip(sub_prob, -1)
                if(down_sample_rate != 1.0):
                    sub_prob = ndimage.interpolation.zoom(sub_prob, 1.0/down_sample_rate, order = 0)  
                prob_batch.append([sub_prob])                
        data_batch = np.asarray(data_batch, np.float32)
        weight_batch = np.asarray(weight_batch, np.float32)
        label_batch = np.asarray(label_batch, np.int64)
        prob_batch = np.asarray(prob_batch, np.float32)
        batch = {}
        batch['images']  = np.transpose(data_batch, [0, 2, 3, 4, 1])
        batch['weights'] = np.transpose(weight_batch, [0, 2, 3, 4, 1])
        batch['labels']  = np.transpose(label_batch, [0, 2, 3, 4, 1])
        if(self.with_probability):
            batch['probs']   = np.transpose(prob_batch, [0, 2, 3, 4, 1])
        else:
            batch['probs']  = None
        return batch
    
    # The following two functions are used for testing
    def get_total_image_number(self):
        return len(self.data)
    
    def get_image_data_with_name(self, i, with_probability = False):
        if(with_probability):
            return [self.data[i], self.weight[i], self.prob[i], self.patient_names[i]]
        else:
            return [self.data[i], self.weight[i], self.patient_names[i]]
    
if __name__ == "__main__":
    loader = BratsLoader()
    loader.load_data("test")
    random.seed(1)
    batch = loader.get_subimage_batch()
    data_batch = np.transpose(batch['images'],[0, 4, 1, 2, 3])
    label_batch = np.asarray(np.transpose(batch['labels'],[0, 4, 1, 2, 3]), np.uint8)
    weight_batch = np.transpose(batch['weights'],[0, 4, 1, 2, 3])
     
    save_folder = '/Users/guotaiwang/Documents/workspace/tf_project/tf_brats/data_process/temp_data'
    for batch_idx in range(5):
        for mod in range(4):     
            save_name = os.path.join(save_folder, "batch_{0:}_{1:}.nii.gz".format(batch_idx, mod))
            save_array_as_nifty_volume(data_batch[batch_idx][mod], save_name)
        save_name = os.path.join(save_folder, "label_{0:}.nii.gz".format(batch_idx))
        save_array_as_nifty_volume(label_batch[batch_idx][0], save_name)
        save_name = os.path.join(save_folder, "weight_{0:}.nii.gz".format(batch_idx))
        save_array_as_nifty_volume(weight_batch[batch_idx][0], save_name)
