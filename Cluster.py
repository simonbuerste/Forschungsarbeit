import tensorflow as tf
import numpy as np

from input_fn import input_fn
from evaluation import evaluate_sess
from training import train_and_evaluate
from VAE import vae_model_fn
from keras import backend as K
from kMeans import kmeans_model_fn
from utils import samples_latentspace
from utils import save_dict_to_json
from metrics import cluster_accuracy
from metrics import normalized_mutual_information
from metrics import adjuster_rand_index

from tensorflow.examples.tutorials.mnist import input_data

import os

# Set the random seed for the whole graph for reproductible experiments
tf.set_random_seed(230)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Set Parameters for Data Preparation and Training
params = {
    "batch_size":           10000,
    "buffer_size":          10000,
    "train_size":           5000,
    "eval_size":            25,
    "num_epochs":           1000,
    "save_summary_steps":   100,
    "k":                    25,     # The number of clusters
    "num_classes":          10      # The 10 digits
}

#model_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/'
model_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'models', 'VAE')
#restore_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/best_weights/'
restore_from = 'best_weights'

data_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'MNIST_data')
#data_dir = '/MNIST_data'
mnist = input_data.read_data_sets(data_dir)

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

with tf.Session(config=config) as sess:
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
    sess.run(cluster_model_spec['iterator_init_op'])  # 2nd initialization necessary in Case of batch_size = size_of_data

    num_steps = (params['eval_size'] + params['batch_size'] - 1) // params['batch_size']
    #metrics = evaluate_sess(sess, cluster_model_spec, num_steps)
    metrics_name = '_'.join(restore_from.split('/'))
    #save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
    #save_dict_to_json(metrics, save_path)

    # Clustering
    for i in range(1, params['eval_size'] + 1):
        n_batches = 0
        accuracy = 0
        nmi = 0
        ari = 0
        try:
            while True:
                _, idx, labels = sess.run(
                    [cluster_model_spec['train_op'], cluster_model_spec['cluster_idx'], cluster_inputs["labels"]])
                # Evaluate

                # Assign a label to each centroid
                # Count total number of labels per centroid, using the label of each training
                # sample to their closest centroid (given by 'cluster_idx')
                counts = np.zeros(shape=(params['k'], params['num_classes']))
                for j in range(len(idx)):
                    counts[idx[j], labels[j]] += 1
                counts = tf.convert_to_tensor(counts)

                # Assign the most frequent label to the centroid
                labels_map = tf.argmax(counts, axis=1)  # find Label with max. occurrence along each row

                # Evaluation ops
                # Lookup: centroid_id -> label
                y_pred = tf.nn.embedding_lookup(labels_map, idx)

                accuracy += sess.run(cluster_accuracy(labels, y_pred))
                nmi += sess.run(normalized_mutual_information(counts))
                ari += sess.run(adjuster_rand_index(counts))
                n_batches += 1
        except tf.errors.OutOfRangeError:
            sess.run(cluster_model_spec['iterator_init_op'])
            pass
        print("Test Accuracy:", accuracy / n_batches)
        print("Normalized Mutual Information:", nmi / n_batches)
        print("Adjusted Rand Index:", ari / n_batches)

    metrics = {
        'Accuracy': accuracy / n_batches,
        'NMI': nmi / n_batches,
        'ARI': ari / n_batches
    }

    save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
    save_dict_to_json(metrics, save_path)
        #if i % 10 == 0 or i == 1:




# Evaluate the model
# evaluate(cluster_model_spec, model_dir, params, restore_from)

# train_and_evaluate(train_model_spec, cluster_model_spec, model_dir, params)  # add ", restore_dir" if a restore Direction is available


