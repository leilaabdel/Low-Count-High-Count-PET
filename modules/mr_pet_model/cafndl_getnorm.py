import dicom
import nibabel as nib
import numpy as np
from cafndl_utils import augment_data

def get_norm_from_nifti(path_load, norm_factor, scale_by_norm=True): 
#	global norm_factor
# get nifti
	nib_load = nib.load(path_load)
	print(nib_load.header)
	# get data
	data_load = nib_load.get_data()
	# transpose to slice*x*y*channel
	data_load = np.transpose(data_load[:,:,:,np.newaxis], [2,0,1,3])
	# scale
	if scale_by_norm:
		norm_factor = np.linalg.norm(data_load.flatten())
	return norm_factor