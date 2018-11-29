import tensorflow as tf

from input_fn import input_fn
from training import train_and_evaluate
from VAE import vae_model_fn
from keras import backend as K
from kMeans import kmeans_model_fn

from tensorflow.examples.tutorials.mnist import input_data

import os

# Set the random seed for the whole graph for reproductible experiments
tf.set_random_seed(230)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Set Parameters for Data Preparation and Training
params = {
    "batch_size":           64,
    "buffer_size":          50000,
    "train_size":           50000,
    "eval_size":            10,
    "num_epochs":           50,
    "save_summary_steps":   10,
    "k":                    25,     # The number of clusters
    "num_classes":          10      # The 10 digits
}

model_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'models', 'VAE')
#restore_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/'

data_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'MNIST_data')
#data_dir = '/MNIST_data'
mnist = input_data.read_data_sets(data_dir)

# Creates an iterator and a dataset
train_inputs = input_fn('train', mnist.train, params)
cluster_inputs = input_fn('cluster', mnist.test, params)

# Define the models (2 different set of nodes that share weights for train and eval)
train_model_spec = vae_model_fn('train', train_inputs, params)
cluster_model_spec = vae_model_fn('cluster', cluster_inputs, params, reuse=True)

# Train the model
train_and_evaluate(train_model_spec, cluster_model_spec, model_dir, params, config)  # add ", restore_dir" if a restore Dir                                           