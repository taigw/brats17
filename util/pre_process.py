import os    
import nibabel
import numpy as np
import random


def get_roi_size(inputVolume):
    [d_idxes, h_idxes, w_idxes] = np.nonzero(inputVolume)
    mind = d_idxes.min(); maxd = d_idxes.max()
    minh = h_idxes.min(); maxh = h_idxes.max()
    minw = w_idxes.min(); maxw = w_idxes.max()
    return [maxd - mind, maxh - minh, maxw - minw]

def get_unique_image_name(img_name_list, subname):
    img_name = [x for x in img_name_list if subname in x]
    assert(len(img_name) == 1)
    return img_name[0]

def load_nifty_volume_as_array(filename):
    img = nibabel.load(filename)
    return img.get_data()

def load_all_modalities_in_one_folder(patient_dir):
    img_name_list = os.listdir(patient_dir)
    img_list = []
    sub_name_list = ['flair.nii', 't1ce.nii', 't1.nii', 't2.nii', 'seg.nii']
    for sub_name in sub_name_list:
        img_name = get_unique_image_name(img_name_list, sub_name)
        img   = load_nifty_volume_as_array(os.path.join(patient_dir, img_name))
        img_list.append(img)
    return img_list

def get_itensity_statistics(volume, n_pxl, iten_sum, iten_sq_sum):
    volume = np.asanyarray(volume, np.float32)
    pixels = volume[volume > 0]
    n_pxl = n_pxl + len(pixels)
    iten_sum = iten_sum + pixels.sum()
    iten_sq_sum = iten_sq_sum + np.square(pixels).sum()
    return n_pxl, iten_sum, iten_sq_sum

def get_all_patients_dir(data_root):
    sub_sets = ['HGG/', 'LGG/']
    all_patients_list = []
    for sub_source in sub_sets:
        sub_source = data_root + sub_source
        patient_list = os.listdir(sub_source)
        patient_list = [sub_source + x for x in patient_list if 'Brats' in x]
        all_patients_list.extend(patient_list)
        print('patients for ', sub_source,len(patient_list))
    print("total patients ", len(all_patients_list))
    return all_patients_list

def get_roi_range_in_one_dimention(x0, x1, L):
    margin = L - (x1 - x0)
    mg0 = margin/2
    mg1 = margin - mg0
    x0 = x0 - mg0
    x1 = x1 + mg1
    return [x0, x1]

def get_roi_from_volumes(volumes):
    [outD, outH, outW] = [144, 176, 144]
    [d_idxes, h_idxes, w_idxes] = np.nonzero(volumes[0])
    mind = d_idxes.min(); maxd = d_idxes.max()
    minh = h_idxes.min(); maxh = h_idxes.max()
    minw = w_idxes.min(); maxw = w_idxes.max()
    print(mind, maxd, minh, maxh, minw, maxw)
    [mind, maxd] = get_roi_range_in_one_dimention(mind, maxd, outD)
    [minh, maxh] = get_roi_range_in_one_dimention(minh, maxh, outH)
    [minw, maxw] = get_roi_range_in_one_dimention(minw, maxw, outW)
    print(mind, maxd, minh, maxh, minw, maxw)
    roi_volumes = []
    for volume in volumes:
        roi_volume = volume[np.ix_(range(mind, maxd), range(minh, maxh), range(minw, maxw))]
        roi_volumes.append(roi_volume)
        print(roi_volume.shape)
    return roi_volumes, [mind, maxd, minh, maxh, minw, maxw]

def get_training_set_statistics(): 
    source_root = '/Users/guotaiwang/Documents/data/BRATS2017/BRATS17TrainingData/'
    all_patients_list = get_all_patients_dir(source_root)

    # get itensity mean and std
#     n_pxls = np.zeros([4], np.float32)
#     iten_sum = np.zeros([4], np.float32)
#     iten_sq_sum = np.zeros([4], np.float32)
#     for patient_dir in all_patients_list:
#         volumes = load_all_modalities_in_one_folder(patient_dir)
#         for i in range(4):
#             n_pxls[i], iten_sum[i], iten_sq_sum[i] = get_itensity_statistics(
#                     volumes[i], n_pxls[i], iten_sum[i], iten_sq_sum[i])
#         print patient_dir
#         print volumes[0][volumes[0]>0].mean(), volumes[1][volumes[1]>0].mean(), volumes[2][volumes[2]>0].mean(), volumes[3][volumes[3]>0].mean()
#     mean = np.divide(iten_sum, n_pxls)
#     sq_men = np.divide(iten_sq_sum, n_pxls)
#     std = np.sqrt(sq_men - np.square(mean))
#     print mean, std    
   
    roi_size = []
    for patient_dir in all_patients_list:
        volumes = load_all_modalities_in_one_folder(patient_dir)
        for i in range(4):
            roi = get_roi_size(volumes[i])
            roi_size.append(roi)
    roi_size = np.asarray(roi_size)
    print(roi_size.mean(axis = 0), roi_size.std(axis = 0))
        
def extract_roi_for_training_set():
    source_root = '/Users/guotaiwang/Documents/data/BRATS2017/BRATS17TrainingData/'
    target_root = 'Training_extract'
    sub_sets = ['HGG/', 'LGG/']
    modality_names = ['flair.nii.gz', 't1ce.nii.gz', 't1.nii.gz', 't2.nii.gz', 'seg.nii.gz']
    all_patients_list = get_all_patients_dir(source_root)
    for patient_dir in all_patients_list:
        volumes = load_all_modalities_in_one_folder(patient_dir)
        roi_volumes, roi = get_roi_from_volumes(volumes)
        for i in range(len(roi_volumes)):
            save_patient_dir = patient_dir.replace("BRATS17TrainingData", target_root)
            print(save_patient_dir)
            if(not os.path.isdir(save_patient_dir)):
                os.mkdir(save_patient_dir)
            save_name = os.path.join(save_patient_dir, modality_names[i])
            img = nibabel.Nifti1Image(roi_volumes[i], np.eye(4))
            nibabel.save(img, save_name)
            
def split_data(split_name, seed):
    source_root = '/Users/guotaiwang/Documents/data/BRATS2017/Training_extract/'
    all_patients_list =  get_all_patients_dir(source_root) 
    random.seed(seed)
    n = len(all_patients_list)
    n_test = 50
    test_mask = np.zeros([n])
    test_idx = random.sample(range(n), n_test)
    test_mask[test_idx] = 1
    
    train_list = []  
    test_list  = [] 
    for i in range(n): 
        patient_split = all_patients_list[i].split('/')
        patient = patient_split[-2] + '/' + patient_split[-1]
        if(test_mask[i]):
            test_list.append(patient) 
        else:
            train_list.append(patient)
    print("train_list", len(train_list))
    print("test_list ", len(test_list))
    train_file = open(split_name + '/train.txt', 'w')
    for patient in train_list:
        train_file.write("%s\n" % patient)
    test_file = open(split_name + '/test.txt', 'w')
    for patient in test_list:
        test_file.write("%s\n" % patient)  
    seed_file =  open(split_name + '/seed.txt', 'w')
    seed_file.write("%d"%seed)
if __name__ == "__main__":
#     get_training_set_statistics()
#     extract_roi_for_training_set()
    split_data('split1', 0)
