from tabnanny import verbose
import pydicom
import nibabel as nib
import numpy as np
from cafndl_utils import augment_data
import pickle
import glob 
import os

def generate_file_list_object(filtered_patient_list_path, input_data_root, trial_time="3600-180", recon_alg="OP"):
    list_dataset_train = []

    with open(filtered_patient_list_path, "rb") as f:
        gt_patient_list = pickle.load(f)

    current_patients = [pat.split("/")[-2]  for pat in glob.glob(f"{input_data_root}/**/")]

    for patient in gt_patient_list:
        if patient in current_patients:
            list_dataset_train.append({
            'input' : [ f"{input_data_root}/{patient}/pet_nifti/{trial_time}_{recon_alg}.nii.gz", 
            f"{input_data_root}/{patient}/mr_nifti/t1_img_registered.nii.gz"
            ],
            'gt':f"{input_data_root}/{patient}/pet_nifti/gt_recon.nii.gz"
            })

    return list_dataset_train


def prepare_data_from_nifti(path_load, list_augments=[], scale_by_norm=True):
	# get nifti
	nib_load = nib.load(path_load)
	print(nib_load.header)
	# get data
	data_load = np.squeeze(nib_load.get_fdata())
	# transpose to slice*x*y*channel
	print("SHAPE", data_load.shape)
	data_load = np.transpose(data_load[:,:,:,np.newaxis], [2,0,1,3])
	# scale
	if scale_by_norm:
		#df = data_load.flatten()
		#norm_factor = np.linalg.norm(df)
		data_load = data_load / np.linalg.norm(data_load.flatten())
	# finish loading data
	print('loaded from {0}, data size {1} (sample, x, y, channel)'.format(path_load, data_load.shape))    
	
	# augmentation
	if len(list_augments)>0:
		print('data augmentation')
		list_data = []
		for augment in list_augments:
			print(augment)
			data_augmented = augment_data(data_load, axis_xy = [1,2], augment = augment)
			list_data.append(data_augmented.reshape(data_load.shape))
		data_load = np.concatenate(list_data, axis = 0)
	return data_load #, norm_factor # KC 20171018


def generateNiiFromImageObject(model,data_train_input, out_path):
	
	# for slice in data_train_input:
		high_count_pred = model.predict(data_train_input, verbose=1)
		high_count_pred = np.squeeze(np.swapaxes(high_count_pred, 0, 2))
		high_count_pred = np.flip(high_count_pred, axis=1)
		high_count_pred = np.rot90(high_count_pred)
		print("SHAPE_FINAL", high_count_pred.shape)
		high_count_pred_nii = nib.Nifti1Image(high_count_pred, affine=np.eye(4))
		nib.save(high_count_pred_nii, out_path)

def generate_masked_nii(input_img_path, mask_img_path, out_path):
	input_img = nib.load(input_img_path).get_fdata()
	mask_img = nib.load(mask_img_path).get_fdata()
	masked_img = input_img * mask_img
	masked_img_nii = nib.Nifti1Image(masked_img, affine=np.eye(4))
	nib.save(masked_img_nii, out_path)
