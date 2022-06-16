#imports
from ast import arg
from operator import index
import nibabel as nib
import numpy as np
import os 
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


import os
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from cafndl_network import deepEncoderDecoder
from cafndl_fileio import prepare_data_from_nifti, generate_file_list_object, generateNiiFromImageObject

#paths
USER = "leila"
CHECKPOINT_NAME = f"test_3_samp_5_epoch.ckpt"
CHECKPOINT_PATH = f"/autofs/space/celer_001/users/{USER}/ckpt/{CHECKPOINT_NAME}"
TRACER = "pbr28"
FINAL_DATA_PATH_FOR_MODEL = f"/autofs/space/celer_001/users/{USER}/data/{TRACER}"
FILTERED_PATIENT_LIST_WITH_GT = f"/autofs/space/celer_001/users/{USER}/working_{TRACER}/pickles/unfiltered_patient_list.pkl"


list_dataset_train = generate_file_list_object(FILTERED_PATIENT_LIST_WITH_GT , FINAL_DATA_PATH_FOR_MODEL)


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=False,
	help="path to weights file")
args = vars(ap.parse_args())

'''
augmentation
'''
list_augments = []
num_augment=len(list_augments)
print('will augment data with {0} augmentations'.format(num_augment))

weights = args['weights']

num_dataset_train = len(list_dataset_train)

final_train_shape = []

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
num_channel_input = 2
num_channel_output = 1
img_rows = 344
img_cols = 344
keras_memory = 0.4
keras_backend = 'tf'
with_batch_norm = True
print('setup parameters')


'''
init model
'''
model = deepEncoderDecoder(num_channel_input = num_channel_input,
                        num_channel_output = num_channel_output,
                        img_rows = img_rows,
                        img_cols = img_cols,
                        lr_init = lr_init, 
                        num_poolings = num_poolings, 
                        num_conv_per_pooling = num_conv_per_pooling, 
                        with_bn = with_batch_norm, verbose=1)
model.load_weights(weights)

for index_data in range(num_dataset_train):
    print("INDEX" , list_dataset_train[index_data]['input'])
    pet_save_path = os.path.join(os.path.dirname(list_dataset_train[index_data]['input'][0]) , f"predicted_{os.path.basename(weights)[:-5]}.nii.gz")
    headmask = prepare_data_from_nifti(os.path.dirname(list_dataset_train[index_data]['input'][1])+'/t1_mask_registered.nii.gz', list_augments, False)

    list_data_train_input = []
    for path_train_input in list_dataset_train[index_data]['input']:
        # load data
        data_train_input = prepare_data_from_nifti(path_train_input, list_augments)
        data_train_input = np.multiply(data_train_input, headmask) # data_train_input 
        list_data_train_input.append(data_train_input)
    data_train_input = np.concatenate(list_data_train_input, axis=-1)

    path_train_gt = list_dataset_train[index_data]['gt']
    data_train_gt = prepare_data_from_nifti(path_train_gt, list_augments)
    
    generateNiiFromImageObject(model, data_train_input, pet_save_path)







    
