import os    
import nibabel
import numpy as np
import random
from scipy import ndimage

def get_roi(temp_label, margin):
    [d_idxes, h_idxes, w_idxes] = np.nonzero(temp_label)
    [D, H, W] = temp_label.shape
    mind = max(d_idxes.min() - margin, 0)
    maxd = min(d_idxes.max() + margin, D)
    minh = max(h_idxes.min() - margin, 0)
    maxh = min(h_idxes.max() + margin, H)
    minw = max(w_idxes.min() - margin, 0)
    maxw = min(w_idxes.max() + margin, W)   
    return [mind, maxd, minh, maxh, minw, maxw]

def get_largest_two_component(img, prt = False, threshold = None):
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(prt):
        print('component size', sizes_list)
    if(len(sizes) == 1):
        return img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:    
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(prt):
                print(max_size2, max_size1, max_size2/max_size1)   
            if(max_size2*10 > max_size1):
                component1 = (component1 + component2) > 0
            
            return component1

def fill_holes(img): 
    neg = 1 - img
    s = ndimage.generate_binary_structure(3,1) # iterate structure
    labeled_array, numpatches = ndimage.label(neg,s) # labeling
    sizes = ndimage.sum(neg,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    max_size = sizes_list[-1]
    max_label = np.where(sizes == max_size)[0] + 1
    component = labeled_array == max_label
    return 1 - component


def remove_external_core(lab_main, lab_ext):
    # for each component of lab_ext, compute the overlap with lab_main
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(lab_ext,s) # labeling
    sizes = ndimage.sum(lab_ext,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    new_lab_ext = np.zeros_like(lab_ext)
    for i in range(len(sizes)):
        sizei = sizes_list[i]
        labeli =  np.where(sizes == sizei)[0] + 1
        componenti = labeled_array == labeli
        overlap = componenti * lab_main
        if((overlap.sum()+ 0.0)/sizei >= 0.5):
            new_lab_ext = np.maximum(new_lab_ext, componenti)
    return new_lab_ext


def get_unique_image_name(img_name_list, subname):
    img_name = [x for x in img_name_list if subname in x]
    assert(len(img_name) == 1)
    return img_name[0]

def load_all_modalities_in_one_folder(patient_dir):
    img_name_list = os.listdir(patient_dir)
    img_list = []
    sub_name_list = ['flair.nii', 't1ce.nii', 't1.nii', 't2.nii', 'seg.nii']
    for sub_name in sub_name_list:
        img_name = get_unique_image_name(img_name_list, sub_name)
        img   = load_nifty_volume_as_array(os.path.join(patient_dir, img_name))
        img_list.append(img)
    return img_list




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
