
# coding: utf-8

# In[ ]:


#%export CUDA_VISIBLE_DEVICES=0
# %load script_demo_train.py
from scipy import io as sio
import numpy as np
import os
from cafndl_fileio import prepare_data_from_nifti, generate_file_list_object
import nibabel as nib
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
from cafndl_fileio import *
from cafndl_utils import *
from cafndl_network import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import argparse

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# exit()

'''
convert dicom to nifti
'''

ckpt_id = sys.argv[1]
'''
dataset
'''

# filename_checkpoint = '../ckpt/'+ckpt_id+'.'  # model_demo_1130_ASL_set1
filename_checkpoint = os.path.join('../ckpt' , "weights-{epoch:03d}-{val_loss:.4f}.hdf5")
filename_init = ''
USER = "leila"
TRACER = "pbr28"
FINAL_DATA_PATH_FOR_MODEL = f"/autofs/space/celer_001/users/{USER}/data/{TRACER}"
FILTERED_PATIENT_LIST_WITH_GT = f"/autofs/space/celer_001/users/{USER}/working_{TRACER}/pickles/unfiltered_patient_list.pkl"


list_dataset_train = generate_file_list_object(FILTERED_PATIENT_LIST_WITH_GT , FINAL_DATA_PATH_FOR_MODEL)

## Generate the training dataset 
# list_dataset_train =  [
# 				{ #4
# 				 'input':[f"{FINAL_DATA_PATH_FOR_MODEL}/PBRKOA_HC021_01/pet_nifti/3600-180_OP.nii.gz",
# 				 		  f'{FINAL_DATA_PATH_FOR_MODEL}/PBRKOA_HC021_01/mr_nifti/t1_img_registered.nii.gz',
# #				 		  '/data3/Amyloid/1350/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		#   '/data3/Amyloid/1350/mr_nifti/T2_nifti_inv.nii',
# 				 		#   '/data3/Amyloid/1350/mr_nifti/T2_FLAIR_nifti_inv.nii'
# 						   ],
# 				 'gt':f"{FINAL_DATA_PATH_FOR_MODEL}/PBRKOA_HC021_01/pet_nifti/gt_recon.nii.gz"
# 				},
# 				{ #5
# 				 'input':['/data3/Amyloid/1355/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1355/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1355/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1355/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1355/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1355/pet_nifti/500_.nii.gz'
# 				},
# 				{ #4
# 				 'input':['/data3/Amyloid/1726/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1726/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1726/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1726/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1726/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1726/pet_nifti/500_.nii.gz'
# 				},
# 				{ #5
# 				 'input':['/data3/Amyloid/1732/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1732/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1732/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1732/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1732/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1732/pet_nifti/500_.nii.gz'
# 				},
# 				{ #2
# 				 'input':['/data3/Amyloid/1750/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1750/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1750/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1750/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1750/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1750/pet_nifti/500_.nii.gz'
# 				},
# 				{ #2
# 				 'input':['/data3/Amyloid/1758/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1758/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1758/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1758/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1758/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1758/pet_nifti/500_.nii.gz'
# 				},
# #				{ #1
# #				 'input':['/data3/Amyloid/1762/pet_nifti/501_.nii.gz',
# #				 		  '/data3/Amyloid/1762/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1762/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# #				 		  '/data3/Amyloid/1762/mr_nifti/T2_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1762/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# #				 'gt':'/data3/Amyloid/1762/pet_nifti/500_.nii.gz'
# #				},
# #				{ #1
# #				 'input':['/data3/Amyloid/1785/pet_nifti/501_.nii.gz',
# #				 		  '/data3/Amyloid/1785/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1785/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# #				 		  '/data3/Amyloid/1785/mr_nifti/T2_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1785/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# #				 'gt':'/data3/Amyloid/1785/pet_nifti/500_.nii.gz'
# #				},
# #				{ #1 NEW
# #				 'input':['/data3/Amyloid/1791/pet_nifti/501_.nii.gz',
# #				 		  '/data3/Amyloid/1791/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1791/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# #				 		  '/data3/Amyloid/1791/mr_nifti/T2_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1791/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# #				 'gt':'/data3/Amyloid/1791/pet_nifti/500_.nii.gz'
# #				},
# 				{ #3
# 				 'input':['/data3/Amyloid/1827/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1827/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1827/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1827/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1827/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1827/pet_nifti/500_.nii.gz'
# 				},
# 				{ #3
# 				 'input':['/data3/Amyloid/1838/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1838/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1838/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1838/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1838/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1838/pet_nifti/500_.nii.gz'
# 				},
# 				{ #3
# 				 'input':['/data3/Amyloid/1905/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1905/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1905/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1905/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1905/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1905/pet_nifti/500_.nii.gz'
# 				},
# 				{ #3
# 				 'input':['/data3/Amyloid/1907/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1907/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1907/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1907/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1907/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1907/pet_nifti/500_.nii.gz'
# 				},
# 				{ #5
# 				 'input':['/data3/Amyloid/1947/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1947/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1947/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1947/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1947/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1947/pet_nifti/500_.nii.gz'
# 				},
# 				{ #4
# 				 'input':['/data3/Amyloid/1978/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1978/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1978/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1978/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1978/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1978/pet_nifti/500_.nii.gz'
# 				},
# #				{ #1
# #				 'input':['/data3/Amyloid/2014/pet_nifti/501_.nii.gz',
# #				 		  '/data3/Amyloid/2014/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2014/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# #				 		  '/data3/Amyloid/2014/mr_nifti/T2_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2014/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# #				 'gt':'/data3/Amyloid/2014/pet_nifti/500_.nii.gz'
# #				},
# 				{ #4
# 				 'input':['/data3/Amyloid/2016/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2016/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2016/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2016/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2016/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2016/pet_nifti/500_.nii.gz'
# 				},
# 				{ #2 NEW
# 				 'input':['/data3/Amyloid/2157/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2157/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2157/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2157/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2157/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2157/pet_nifti/500_.nii.gz'
# 				},
# 				{ #2 NEW
# 				 'input':['/data3/Amyloid/2214/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2214/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2214/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2214/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2214/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2214/pet_nifti/500_.nii.gz'
# 				},
# 				{ #3 NEW
# 				 'input':['/data3/Amyloid/2304/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2304/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2304/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2304/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2304/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2427/pet_nifti/500_.nii.gz'
# 				},
# #				{ #1 NEW (orig3)
# #				 'input':['/data3/Amyloid/2317/pet_nifti/501_.nii.gz',
# #				 		  '/data3/Amyloid/2317/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2317/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# #				 		  '/data3/Amyloid/2317/mr_nifti/T2_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2317/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# #				 'gt':'/data3/Amyloid/2317/pet_nifti/500_.nii.gz'
# #				},
# 				{ #2 NEW (orig4)
# 				 'input':['/data3/Amyloid/2376/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2376/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2376/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2376/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2376/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2376/pet_nifti/500_.nii.gz'
# 				},
# 				{ #4 NEW
# 				 'input':['/data3/Amyloid/2427/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2427/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2427/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2427/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2427/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2427/pet_nifti/500_.nii.gz'
# 				},
# 				{ #5 NEW
# 				 'input':['/data3/Amyloid/2482/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2482/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2482/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2482/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2482/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2482/pet_nifti/500_.nii.gz'
# 				},
# 				{ #5 NEW
# 				 'input':['/data3/Amyloid/2516/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2516/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2516/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2516/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2516/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2516/pet_nifti/500_.nii.gz'
# 				},
# #				{ #1 NEW
# #				 'input':['/data3/Amyloid/2414/pet_nifti/501_.nii.gz',
# #				 		  '/data3/Amyloid/2414/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2414/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# #				 		  '/data3/Amyloid/2414/mr_nifti/T2_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2414/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# #				 'gt':'/data3/Amyloid/2414/pet_nifti/500_.nii.gz'
# #				},
# 				{ #2 NEW
# 				 'input':['/data3/Amyloid/1961/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1961/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1961/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1961/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1961/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1961/pet_nifti/500_.nii.gz'
# 				},
# 				{ #3 NEW
# 				 'input':['/data3/Amyloid/2185/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2185/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2185/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2185/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2185/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2185/pet_nifti/500_.nii.gz'
# 				},
# 				{ #4 NEW
# 				 'input':['/data3/Amyloid/2152/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2152/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2152/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2152/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2152/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2152/pet_nifti/500_.nii.gz'
# 				},
# 				{ #5 NEW
# 				 'input':['/data3/Amyloid/2063/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2063/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2063/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2063/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2063/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2063/pet_nifti/500_.nii.gz'
# 				},
# #				{ #1 NEW
# #				 'input':['/data3/Amyloid/1375/pet_nifti/501_.nii.gz',
# #				 		  '/data3/Amyloid/1375/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1375/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# #				 		  '/data3/Amyloid/1375/mr_nifti/T2_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1375/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# #				 'gt':'/data3/Amyloid/1375/pet_nifti/500_.nii.gz'
# #				},
# 				{ #2 NEW
# 				 'input':['/data3/Amyloid/1789/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1789/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1789/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1789/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1789/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1789/pet_nifti/500_.nii.gz'
# 				},
# 				{ #3 NEW
# 				 'input':['/data3/Amyloid/1816/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1816/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1816/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1816/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1816/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1816/pet_nifti/500_.nii.gz'
# 				},
# 				{ #4 NEW
# 				 'input':['/data3/Amyloid/1965/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1965/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1965/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1965/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1965/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1965/pet_nifti/500_.nii.gz'
# 				},
# 				{ #5 NEW
# 				 'input':['/data3/Amyloid/1923/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/1923/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/1923/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/1923/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/1923/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/1923/pet_nifti/500_.nii.gz'
# 				},
# #				{ #1 NEW
# #				 'input':['/data3/Amyloid/2314/pet_nifti/501_.nii.gz',
# #				 		  '/data3/Amyloid/2314/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2314/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# #				 		  '/data3/Amyloid/2314/mr_nifti/T2_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2314/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# #				 'gt':'/data3/Amyloid/2314/pet_nifti/500_.nii.gz'
# #				},
# 				{ #2 NEW
# 				 'input':['/data3/Amyloid/2511/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2511/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2511/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2511/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2511/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2511/pet_nifti/500_.nii.gz'
# 				},
# 				{ #3 NEW
# 				 'input':['/data3/Amyloid/2416/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2416/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2416/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2416/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2416/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2416/pet_nifti/500_.nii.gz'
# 				},
# 				{ #4 NEW
# 				 'input':['/data3/Amyloid/2425/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/2425/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/2425/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/2425/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/2425/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/2425/pet_nifti/500_.nii.gz'
# 				},
# 				{ #5 NEW
# 				 'input':['/data3/Amyloid/50767/pet_nifti/501_.nii.gz',
# 				 		  '/data3/Amyloid/50767/mr_nifti/T1_nifti_inv.nii',
# #				 		  '/data3/Amyloid/50767/mr_nifti/ASL_CBF_nifti_inv.nii.gz',
# 				 		  '/data3/Amyloid/50767/mr_nifti/T2_nifti_inv.nii',
# 				 		  '/data3/Amyloid/50767/mr_nifti/T2_FLAIR_nifti_inv.nii'],
# 				 'gt':'/data3/Amyloid/50767/pet_nifti/500_.nii.gz'
# 				}
				# ] 

dir_train_histroy = '../ckpt/'
num_dataset_train = len(list_dataset_train)                
print('process {0} data description'.format(num_dataset_train))

'''
augmentation
'''
list_augments = []
num_augment_flipxy = 2
num_augment_flipx = 2
num_augment_flipy = 2
num_augment_shiftx = 1
num_augment_shifty = 1
for flipxy in range(num_augment_flipxy):
	for flipx in range(num_augment_flipx):
		for flipy in range(num_augment_flipy):
			for shiftx in range(num_augment_shiftx):
				for shifty in range(num_augment_shifty):
					augment={'flipxy':flipxy,'flipx':flipx,'flipy':flipy,'shiftx':shiftx,'shifty':shifty}
					list_augments.append(augment)
num_augment=len(list_augments)
print('will augment data with {0} augmentations'.format(num_augment))

'''
file loading related 
'''
ext_dicom = 'MRDC'
key_sort = lambda x: int(x.split('.')[-1])
scale_method = lambda x:np.mean(np.abs(x))
scale_by_mean = False
scale_factor = 1/32768.
ext_data = 'npz'
dir_samples = '../data/data_sample/'

def export_data_to_npz(data_train_input, data_train_gt,dir_numpy_compressed, index_sample_total=0, ext_data = 'npz'):  
	index_sample_accumuated = index_sample_total
	num_sample_in_data = data_train_input.shape[0]
	if not os.path.exists(dir_numpy_compressed):
		os.makedirs(dir_numpy_compressed, exist_ok=True)
		print('create directory {0}'.format(dir_numpy_compressed))
	print('start to export data dimension {0}->{1} to {2} for index {3}', 
		  data_train_input.shape, data_train_gt.shape, dir_numpy_compressed, 
		  index_sample_total)        
	for i in range(num_sample_in_data):
		im_input = data_train_input[i,:]
		im_output = data_train_gt[i,:]
		filepath_npz = os.path.join(dir_numpy_compressed,'{0}.{1}'.format(index_sample_accumuated, ext_data))
		with open(filepath_npz,'wb') as file_input:
			np.savez_compressed(file_input, input=im_input, output=im_output)
		index_sample_accumuated+=1
	print('exported data dimension {0}->{1} to {2} for index {3}', 
		  data_train_input.shape, data_train_gt.shape, dir_numpy_compressed, 
		  [index_sample_total,index_sample_accumuated])
	return index_sample_accumuated

'''
generate train data
'''
list_train_input = []
list_train_gt = []  
index_sample_total = 0      
for index_data in range(num_dataset_train):
	# directory
	# headmask = prepare_data_from_nifti(os.path.dirname(list_dataset_train[index_data]['input'][1])+'/headmask_inv.nii', list_augments, False)
	headmask = prepare_data_from_nifti(os.path.dirname(list_dataset_train[index_data]['input'][1])+'/t1_mask_registered.nii.gz', list_augments, False)

	list_data_train_input = []
	for path_train_input in list_dataset_train[index_data]['input']:
		# load data
		data_train_input = prepare_data_from_nifti(path_train_input, list_augments)
		data_train_input = np.multiply(data_train_input, headmask) # data_train_input 
		list_data_train_input.append(data_train_input)
	data_train_input = np.concatenate(list_data_train_input, axis=-1)
	
	
	# load data ground truth
	path_train_gt = list_dataset_train[index_data]['gt']
	data_train_gt = prepare_data_from_nifti(path_train_gt, list_augments)
	# data_train_gt = np.multiply(data_train_gt, headmask) # REMOVE HEADMASK FOR NOW
	#data_train_residual = data_train_gt - data_train_input[:,:,:,0:1] # changed!
	# append
	# list_train_input.append(data_train_input)
	# list_train_gt.append(data_train_gt)

	# export
	index_sample_total = export_data_to_npz(data_train_input, 
											data_train_gt, 
											dir_samples, 
											index_sample_total, 
											ext_data)


'''
setup parameters
'''
# related to model
num_poolings = 3
num_conv_per_pooling = 3
# related to training
lr_init = 0.0002
num_epoch = 100
ratio_validation = 0.1
validation_split = 0.1
batch_size = 4
y_range = [-0.5,0.5]
# default settings
num_channel_input = data_train_input.shape[-1]
num_channel_output = data_train_gt.shape[-1]
img_rows = data_train_input.shape[1]
img_cols = data_train_gt.shape[1]
keras_memory = 0.4
keras_backend = 'tf'
with_batch_norm = True
print('setup parameters')


'''
init model
'''
callback_checkpoint = ModelCheckpoint(filename_checkpoint, 
								monitor='val_loss', 
								save_best_only=True, save_weights_only=True, save_freq='epoch')
setKerasMemory(keras_memory)
model = deepEncoderDecoder(num_channel_input = num_channel_input,
						num_channel_output = num_channel_output,
						img_rows = img_rows,
						img_cols = img_cols,
						lr_init = lr_init, 
						num_poolings = num_poolings, 
						num_conv_per_pooling = num_conv_per_pooling, 
						with_bn = with_batch_norm, verbose=1)
print('train model:', filename_checkpoint)
print('parameter count:', model.count_params())


'''
define generator
'''
# details inside generator
params_generator = {'dim_x': img_rows,
		  'dim_y': img_cols,
		  'dim_z': num_channel_input,
		  'dim_output': num_channel_output,
		  'batch_size': 4,
		  'shuffle': True,
		  'verbose': 0,
		  'scale_data': 100.,
		  'scale_baseline': 1.0}
print('generator parameters:', params_generator)

class DataGenerator(object):
	'Generates data for Keras'
	def __init__(self, dim_x = 512, dim_y = 512, dim_z = 6, dim_output = 1, 
				batch_size = 2, shuffle = True, verbose = 1,
				scale_data = 1.0, scale_baseline = 1.0):
		'Initialization'
		self.dim_x = dim_x
		self.dim_y = dim_y
		self.dim_z = dim_z
		self.dim_output = dim_output
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.verbose = verbose
		self.scale_data = scale_data
		self.scale_baseline = scale_baseline

	def generate(self, dir_sample, list_IDs):
		'Generates batches of samples'
		# Infinite loop
		while 1:
			# Generate order of exploration of dataset
			indexes = self.__get_exploration_order(list_IDs)
			if self.verbose>0:
				print('indexes:', indexes)
			# Generate batches
			imax = int(len(indexes)/self.batch_size)
			if self.verbose>0:            
				print('imax:', imax)
			for i in range(imax):
				# Find list of IDs
				list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
				if self.verbose>0:
					print('list_IDs_temp:', list_IDs_temp)
				# Generate data
				X, Y = self.__data_generation(dir_sample, list_IDs_temp)
				if self.verbose>0:                
					print('generated dataset size:', X.shape, Y.shape)

				yield X, Y

	def __get_exploration_order(self, list_IDs):
		'Generates order of exploration'
		# Find exploration order
		indexes = np.arange(len(list_IDs))
		if self.shuffle == True:
			np.random.shuffle(indexes)
		return indexes

	def __data_generation(self, dir_sample, list_IDs_temp, ext_data = 'npz'):
		'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
		# Initialization
		X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z, 1))
		Y = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_output, 1))

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store volume
			data_load = np.load(os.path.join(dir_sample, '{0}.{1}'.format(ID,ext_data)))
			X[i, :, :, :, 0] = data_load['input']
			Y[i, :, :, :, 0] = data_load['output'] 
		X = X[:,:,:,:,0]
		Y = Y[:,:,:,:,0]        
		X = X * self.scale_data
		Y = Y * self.scale_data
		Y = Y - self.scale_baseline * X[:,:,:,0:1]   
		print(X.shape, Y.shape)   
		return X, Y

''' 
setup train and val generator
'''
validation_split = 0.1
index_sample_total = len([x for x in os.listdir(dir_samples) if x.endswith(ext_data)])
list_indexes_train = np.random.permutation(index_sample_total)
if validation_split>1:
	list_indexes_val = list_indexes_train[-validation_split:].tolist()
	list_indexes_train = list_indexes_train[:int(index_sample_total-validation_split)].tolist()    
else:
	list_indexes_val = list_indexes_train[-int(index_sample_total*validation_split):].tolist()
	list_indexes_train = list_indexes_train[:int(index_sample_total*(1-validation_split))].tolist()
print('train on {0} samples and validation on {1} samples'.format(
		len(list_indexes_train), len(list_indexes_val)))
training_generator = DataGenerator(**params_generator).generate(dir_samples, list_indexes_train)
validation_generator = DataGenerator(**params_generator).generate(dir_samples, list_indexes_val)

'''
sanity check

model.fit(
					generator = training_generator,
					steps_per_epoch = 1,
					epochs = 1,
					validation_data = validation_generator,
					validation_steps = 1,
					max_q_size = 16,
					)		

assert 0
'''

'''
setup learning 
'''
# hyper parameter in each train iteration
#list_hyper_parameters=[{'lr':0.001,'epochs':50},{'lr':0.0002,'epochs':50},{'lr':0.0001,'epochs':30}]
MOD_EPOCHS = 100
list_hyper_parameters=[{'lr':0.0002,'epochs':MOD_EPOCHS}]
type_activation_output = 'linear'


'''
training
'''
index_hyper_start = 0
num_hyper_parameter = len(list_hyper_parameters)
for index_hyper in range(index_hyper_start, num_hyper_parameter):
	hyper_train = dict(list_hyper_parameters[index_hyper])
	print('hyper parameters:', hyper_train)
	# init
	if 'init' in hyper_train:
		try:
			model.load_weights(hyper_train['init'])
		except:
			hyper_train['init'] = ''
			print('failed to learn from init-point ' + hyper_train['init'])
			pass
	else:
		# load previous optimal
		try:
			model.load_weights(filename_checkpoint)
			hyper_train['init'] = filename_checkpoint				
			print('model finetune from ' + filename_checkpoint)   
		except:
			hyper_train['init'] = ''
			print('failed to learn from checkpoint ' + hyper_train['init'])     
			pass
	# update filename and checkpoint
	if 'ckpt' in hyper_train:
		filename_checkpoint = hyper_train['ckpt']
	else:
		hyper_train['ckpt'] = filename_checkpoint
	model_checkpoint =  ModelCheckpoint(filename_checkpoint, 
								monitor='val_loss', 
								save_best_only=True, save_weights_only=True, save_freq='epoch')	

	# update leraning rate
	if 'lr' in hyper_train:
		model.optimizer = Adam(lr=hyper_train['lr'])
	else:
		hyper_train['lr'] = -1 #default

	
	# update epochs
	if 'epochs' in hyper_train:
		epochs = hyper_train['epochs']
	else:
		hyper_train['epochs'] = 50
		epochs = 50

	# update train_list
	hyper_train['list_dataset_train'] = list_dataset_train
	hyper_train['type_activation_output'] = type_activation_output
	hyper_train['y_range'] = np.array(y_range).tolist()		

	# fit data
	t_start_train = datetime.datetime.now()
	
	try:
		# history = model.fit_generator(data_train_input,
		# 				data_train_residual, 
		# 				epochs=epochs, 
		# 				callbacks=[model_checkpoint],
		# 				validation_split=validation_split,
		# 				batch_size=batch_size, 
		# 				shuffle=True, 
		# 				verbose=1)

		print('train with hyper parameters:', hyper_train)
		history = model.fit_generator(
					generator = training_generator,
					steps_per_epoch = len(list_indexes_train)/batch_size,
					epochs = epochs,
					callbacks =[model_checkpoint],
					validation_data = validation_generator,
					validation_steps = len(list_indexes_val)/batch_size,
					# max_q_size = 16,
					)		
	except:
		history = []
		print('break training')
		continue
	t_end_train = datetime.datetime.now()
	t_elapse = (t_end_train-t_start_train).total_seconds()
	print('finish training on {0} samples from data size {1} for {2} epochs using time {3}'.format(
			data_train_input.shape, data_train_input.shape, epochs, t_elapse))
	hyper_train['elapse'] = t_elapse
	
	'''
	save training results
	'''
	# save train loss/val loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	ylim_min = min(min(history.history['loss']), min(history.history['val_loss']))
	ylim_max = max(history.history['loss'])*1.2
	plt.ylim([ylim_min, ylim_max])
	plt.legend(['train', 'test'], loc='upper left')
	path_figure = filename_checkpoint+'_{0}.png'.format(index_hyper)
	plt.savefig(path_figure)

	# save history dictionary
	dict_result = {'history':history.history, 'hyper_parameter':hyper_train}
	path_history = filename_checkpoint + '_{0}.json'.format(index_hyper)
	with open(path_history, 'w') as outfile:
		json.dump(dict_result, outfile)	

