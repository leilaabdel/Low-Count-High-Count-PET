import tensorflow as tf
from cafndl_network import deepEncoderDecoder

CHECKPOINT_PATH = "/autofs/space/celer_001/users/leila/pretrained/mask_0127_set1.ckpt"

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

model.load_weights(CHECKPOINT_PATH, by_name= True, skip_mismatch = True)

