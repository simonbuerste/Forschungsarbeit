import os
import argparse

import tensorflow as tf

from time import gmtime, strftime
from shutil import copyfile

from input_fn import input_fn
from training import train_and_evaluate
from IDEC_Clustering import train_and_evaluate_idec
from utils import Params

from D_VAE1 import vae_model_fn
from D_VAE1_Gumbel import g_vae_model_fn
from Beta_VAE import b_vae_model_fn
from D_AE1 import ae_model_fn
from D_Beta_AE1 import b_ae_model_fn

from kMeans import kmeans_model_fn
from gmm import gmm_model_fn
from IDEC import idec_model_fn
from Argmax import argmax_model_fn

# Set the random seed for the whole graph for reproducible experiments
tf.set_random_seed(230)


config = tf.ConfigProto(inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
config.gpu_options.allow_growth = True

# model_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'Models')
model_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/'

# data_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'Datasets')
data_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Data/'

# restore_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'Models', 'AE_MNIST_kmeans_2019-03-07_14-50', 'best_weights')
restore_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/AE_MNIST_kmeans_2019-03-01_13-18/best_weights'

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
parser.add_argument('--dataset', default='MNIST',
                    help="Choose dataset which should be used")


if __name__ == '__main__':

    # Load the parameters
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, 'params.json')
    params = Params(json_path)

    # get current time as string for saving of model
    timestring = strftime("%Y-%m-%d_%H-%M", gmtime())

    data_dir = os.path.join(data_dir, args.dataset)
    model_dir = os.path.join(model_dir, (args.latent_model + '_' + args.dataset + '_' + args.cluster_model + '_'
                                         + timestring))

    # Create directory for model combination if not existens
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # copy params.json file to the model direction for reproducible results
    copyfile(json_path, os.path.join(model_dir, 'params.json'))

    # Creates an iterator and a dataset
    train_inputs = input_fn(data_dir, 'train', params)
    cluster_inputs = input_fn(data_dir, 'test', params)

    # Define the models (2 different set of nodes that share weights for train and eval)
    if args.latent_model == 'AE':
        train_model_spec = ae_model_fn('train', train_inputs, params)
        cluster_model_spec = ae_model_fn('cluster', cluster_inputs, params, reuse=True)
    elif args.latent_model == 'b_AE':
        train_model_spec = b_ae_model_fn('train', train_inputs, params)
        cluster_model_spec = b_ae_model_fn('cluster', cluster_inputs, params, reuse=True)
    elif args.latent_model == 'VAE':
        train_model_spec = vae_model_fn('train', train_inputs, params)
        cluster_model_spec = vae_model_fn('cluster', cluster_inputs, params, reuse=True)
    elif args.latent_model == 'b_VAE':
        train_model_spec = b_vae_model_fn('train', train_inputs, params)
        cluster_model_spec = b_vae_model_fn('cluster', cluster_inputs, params, reuse=True)
    elif args.latent_model == 'g_VAE':
        train_model_spec = g_vae_model_fn('train', train_inputs, params)
        cluster_model_spec = g_vae_model_fn('cluster', cluster_inputs, params, reuse=True)
    else:
        print("Unknown Model selected")

    vars_to_restore = tf.contrib.framework.get_variables_to_restore()

    # Desired Cluster model is selected
    if args.cluster_model == 'kmeans':
        # Input for Clustering is Output of Encoder
        cluster_inputs["samples"] = cluster_model_spec['sample']
        cluster_model_spec = kmeans_model_fn(cluster_inputs, params)
        # Train the model
        train_and_evaluate(train_model_spec, cluster_model_spec, model_dir, params, config)  # add ", restore_dir" if a restore Dir
    elif args.cluster_model == 'gmm':
        # Input for Clustering is Output of Encoder
        cluster_inputs["samples"] = cluster_model_spec['sample']
        cluster_model_spec = gmm_model_fn(cluster_inputs, params)
        # Train the model
        train_and_evaluate(train_model_spec, cluster_model_spec, model_dir, params, config)  # add ", restore_dir" if a restore Dir
    elif args.cluster_model == 'argmax':
        # Input for Clustering is Output of Encoder
        cluster_inputs["samples"] = cluster_model_spec['sample']
        cluster_model_spec = argmax_model_fn(cluster_inputs, params)
        # Train the model
        train_and_evaluate(train_model_spec, cluster_model_spec, model_dir, params, config)  # add ", restore_dir" if a restore Dir
    elif args.cluster_model == 'IDEC':
        # Input for Clustering is Output of Encoder
        train_inputs["samples"] = train_model_spec['sample']
        train_model_spec = idec_model_fn(train_inputs, train_model_spec, params)
        train_and_evaluate_idec(train_model_spec, model_dir, params, config, restore_dir, vars_to_restore)  # add ", restore_dir" if a restore Dir
    else:
        print("Unknown Model selected")
