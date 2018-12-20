import tensorflow as tf
import os

from input_fn import input_fn
from training import train_and_evaluate
from VAE import vae_model_fn

# Set the random seed for the whole graph for reproducible experiments
tf.set_random_seed(230)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Set Parameters for Data Preparation and Training
params = {
    "batch_size":           64,
    "buffer_size":          50000,
    "train_size":           128,
    "eval_size":            10,
    "num_epochs":           50,
    "save_summary_steps":   10,
    "k":                    25,     # The number of clusters
    "num_classes":          10      # The 10 digits
}

model_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'models', 'VAE')
#model_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/'

data_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'MNIST_data')
#data_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Data/MNIST'

# Creates an iterator and a dataset
train_inputs = input_fn(data_dir, 'train', params)
cluster_inputs = input_fn(data_dir, 'test', params)

# Define the models (2 different set of nodes that share weights for train and eval)
train_model_spec = vae_model_fn('train', train_inputs, params)
cluster_model_spec = vae_model_fn('cluster', cluster_inputs, params, reuse=True)

# Train the model
train_and_evaluate(train_model_spec, cluster_model_spec, model_dir, params, config)  # add ", restore_dir" if a restore Dir
