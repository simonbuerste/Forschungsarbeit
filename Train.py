import os
import argparse

import tensorflow as tf

from input_fn import input_fn
from training import train_and_evaluate
from utils import samples_latentspace
from utils import Params
from VAE import vae_model_fn
from Beta_VAE import b_vae_model_fn
from AE import ae_model_fn
from kMeans import kmeans_model_fn
from gmm import gmm_model_fn

# Set the random seed for the whole graph for reproducible experiments
tf.set_random_seed(230)


config = tf.ConfigProto(inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
config.gpu_options.allow_growth = True

# model_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'models', 'VAE')
model_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/'

# data_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'MNIST_data')
data_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Data/'

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=model_dir,
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default=data_dir,
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")
parser.add_argument('--gpu', default=0,
                    help="Choose GPU on which the program should run")
parser.add_argument('--latent_model', default='AE',
                    help="Choose Model which is used for creating a latent space")
parser.add_argument('--cluster_model', default='kmeans',
                    help="Choose Model which is used for clustering")
parser.add_argument('--dataset', default='CIFAR-100',
                    help="Choose dataset which should be used")


if __name__ == '__main__':

    # Load the parameters
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, 'params.json')
    params = Params(json_path)

    data_dir = os.path.join(data_dir, args.dataset)
    model_dir = os.path.join(model_dir, args.latent_model, args.dataset, args.cluster_model)
    # Create directory for model combination if not existens
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Creates an iterator and a dataset
    train_inputs = input_fn(data_dir, 'train', params)
    cluster_inputs = input_fn(data_dir, 'train', params)

    # Define the models (2 different set of nodes that share weights for train and eval)
    if args.latent_model == 'AE':
        train_model_spec = ae_model_fn('train', train_inputs, params)
        cluster_model_spec = ae_model_fn('cluster', cluster_inputs, params, reuse=True)
    elif args.latent_model == 'VAE':
        train_model_spec = vae_model_fn('train', train_inputs, params)
        cluster_model_spec = vae_model_fn('cluster', cluster_inputs, params, reuse=True)
    elif args.latent_model == 'b_VAE':
        train_model_spec = b_vae_model_fn('train', train_inputs, params)
        cluster_model_spec = b_vae_model_fn('cluster', cluster_inputs, params, reuse=True)
    else:
        print("Unknown Model selected")

    # Input for Clustering is Output of Encoder
    cluster_inputs["img"] = samples_latentspace(cluster_model_spec)

    # Desired Cluster model is selected
    if args.cluster_model == 'kmeans':
        cluster_model_spec = kmeans_model_fn(cluster_inputs, params)
    elif args.cluster_model == 'gmm':
        cluster_model_spec = gmm_model_fn(cluster_inputs, params)

    # Train the model
    train_and_evaluate(train_model_spec, cluster_model_spec, model_dir, params, config)  # add ", restore_dir" if a restore Dir
