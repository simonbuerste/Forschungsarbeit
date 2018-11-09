import tensorflow as tf

import ImportDataset
from Training import train_sess
from VAE import run_vae
from keras import backend as K

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Set Parameters for Data Preparation
batch_size = 64
shuffle_size = 10000
fetch_size = 1
# Set Parameters for Training
no_training_batches = 50
save_model = True

# Import desired Dataset
train_data, test_data, val_data = ImportDataset.importmnist(batch_size, shuffle_size, fetch_size)
# Create One Iterator and initialize with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)    # initializer for test_data

K.set_session(tf.Session(config=config))

# Run Training
loss = train_sess(img, save_model, train_init, no_training_batches)


