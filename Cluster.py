import os
import argparse

import tensorflow as tf
import numpy as np

from input_fn import input_fn
from utils import samples_latentspace
from utils import Params
from evaluation import evaluate

from VAE import vae_model_fn
from AE import ae_model_fn
from kMeans import kmeans_model_fn
from gmm import gmm_model_fn

from tensorflow.examples.tutorials.mnist import input_data


# Set the random seed for the whole graph for reproductible experiments
tf.set_random_seed(230)


config = tf.ConfigProto(inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
config.gpu_options.allow_growth = True

model_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/AE/'
# model_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'models', 'VAE')

# restore_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/best_weights/'
restore_from = 'best_weights'

# data_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'MNIST_data')
data_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Data/MNIST/'

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=model_dir,
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default=data_dir,
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=restore_from,
                    help="Subdirectory of model dir or file containing the weights")
parser.add_argument('--gpu', default=0,
                    help="Choose GPU on which the program should run")

if __name__ == '__main__':

    # Load the parameters
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, 'params.json')
    params = Params(json_path)

    # Creates an iterator and a dataset
    cluster_inputs = input_fn(args.data_dir, 'test', params)

    # Define the model
    latent_model_spec = ae_model_fn('cluster', cluster_inputs, params, reuse=False)
    vars_to_restore = tf.contrib.framework.get_variables_to_restore()

    # Input for kMeans is Output of Encoder
    cluster_inputs["img"] = samples_latentspace(latent_model_spec)
    cluster_model_spec = gmm_model_fn(cluster_inputs, params)

    # Initialize tf.Saver
    saver = tf.train.Saver(vars_to_restore)

    evaluate(cluster_model_spec, args.model_dir, params, args.restore_from, config, saver)
