import tensorflow as tf

from input_fn import input_fn
from evaluation import evaluate_sess
from training import train_and_evaluate
from VAE import vae_model_fn
from keras import backend as K
from kMeans import kmeans_model_fn
from utils import samples_latentspace
from utils import save_dict_to_json
from metrics import cluster_accuracy

from tensorflow.examples.tutorials.mnist import input_data

import os

# Set the random seed for the whole graph for reproductible experiments
tf.set_random_seed(230)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

K.set_session(tf.Session(config=config))

# Set Parameters for Data Preparation and Training
params = {
    "batch_size":           512,
    "buffer_size":          10000,
    "train_size":           5000,
    "eval_size":            10,
    "num_epochs":           1000,
    "save_summary_steps":   100,
    "k":                    25,     # The number of clusters
    "num_classes":          10      # The 10 digits
}

model_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/'
#restore_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/best_weights/'
restore_from = 'best_weights'

mnist = input_data.read_data_sets('/MNIST_data')

# Creates an iterator and a dataset
cluster_inputs = input_fn('cluster', mnist.test, params)

# Define the model
vae_model_spec = vae_model_fn('cluster', cluster_inputs, params, reuse=False)
vars_to_restore = tf.contrib.framework.get_variables_to_restore()

# Input for kMeans is Output of Encoder
cluster_inputs["img"] = samples_latentspace(vae_model_spec)
cluster_model_spec = kmeans_model_fn(cluster_inputs, params)

# Initialize tf.Saver
saver = tf.train.Saver(vars_to_restore)

with tf.Session() as sess:
    # Initialize the lookup table
    sess.run([vae_model_spec['variable_init_op'], cluster_model_spec['variable_init_op']])

    # Reload weights from the weights subdirectory
    save_path = os.path.join(model_dir, restore_from)
    if os.path.isdir(save_path):
        save_path = tf.train.latest_checkpoint(save_path)
    saver.restore(sess, save_path)

    sess.run(cluster_model_spec['iterator_init_op'])
    sess.run(cluster_model_spec['metrics_init_op'])
    sess.run(cluster_model_spec['init_op'])

    num_steps = (params['eval_size'] + params['batch_size'] - 1) // params['batch_size']
    #metrics = evaluate_sess(sess, cluster_model_spec, num_steps)
    #metrics_name = '_'.join(restore_from.split('/'))
    #save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
    #save_dict_to_json(metrics, save_path)

    # Training of Clustering
    for i in range(1, params['eval_size'] + 1):
        _, idx, labels = sess.run([cluster_model_spec['train_op'], cluster_model_spec['cluster_idx'], cluster_inputs["labels"]])
        #if i % 10 == 0 or i == 1:

    # Evaluate
    accuracy = cluster_accuracy(labels, params, idx)
    print("Test Accuracy:", sess.run(accuracy))


# Evaluate the model
# evaluate(cluster_model_spec, model_dir, params, restore_from)

# train_and_evaluate(train_model_spec, cluster_model_spec, model_dir, params)  # add ", restore_dir" if a restore Direction is available


