import pydicom
import nibabel as nib
import numpy as np
from cafndl_utils import augment_data

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