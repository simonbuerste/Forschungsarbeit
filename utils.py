
"""General utility functions"""

import json
import logging
import os
import io
import umap
import numpy as np
import tensorflow as tf

import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def visualize_embeddings(sess, log_dir, writer, params):

    # Get latentspace data and labels from saved files
    sub_latentspace = []
    sub_metadata = []
    for i in range(params.num_epochs):
        if i % params.visualization_step == 0 or i == params.num_epochs - 1:
            metadata = os.path.join(log_dir, ('metadata' + str(i + 1) + '.tsv'))
            img_latentspace = os.path.join(log_dir, ('latentspace' + str(i + 1) + '.txt'))

            latentspace = np.loadtxt(img_latentspace)
            features = tf.Variable(latentspace, name=('latentspace' + str(i+1)))
            sub_latentspace.append(features)
            sub_metadata.append(metadata)

    # Initialize a Saver and variables for embeddings
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    save_path = os.path.join(log_dir, 'latentspace')
    saver.save(sess, save_path)

    # Create a Projector for Tensorboard visualization
    config = projector.ProjectorConfig()
    for i in range(params.num_epochs):
        if i % params.visualization_step == 0 or i == params.num_epochs - 1:
            list_index = i // params.visualization_step
            embedding = config.embeddings.add()
            embedding.tensor_name = sub_latentspace[list_index].name
            embedding.metadata_path = sub_metadata[list_index]

    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(writer, config)


def visualize_umap(sess, log_dir, writer, params):

    # Get latentspace data and labels from saved files
    sub_latentspace = []
    sub_metadata = []
    for i in range(params.num_epochs):
        if i % params.visualization_step == 0 or i == params.num_epochs - 1:
            metadata = os.path.join(log_dir, ('metadata' + str(i + 1) + '.tsv'))
            img_latentspace = os.path.join(log_dir, ('latentspace' + str(i + 1) + '.txt'))

            latentspace = np.loadtxt(img_latentspace)
            sub_latentspace.append(latentspace)
            sub_metadata.append(metadata)

    for i in range(params.num_epochs):
        if i % params.visualization_step == 0 or i == params.num_epochs - 1:
            list_index = i // params.visualization_step

            # Fit UMAP to latentspace data
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
            reducer.fit(sub_latentspace[list_index])
            embedding = reducer.transform(sub_latentspace[list_index])

            # Read Labels from txt
            labels = np.genfromtxt(fname=sub_metadata[list_index], delimiter="\t")

            # Create Scatter Plot with UMAP-transformed data
            f = plt.figure(list_index)
            ax = f.add_subplot(111)
            ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
            ax.set_title(('UMAP projection of latentspace after Epoch %i' % (i+1)), fontsize=14)

            # Create a Buffer and write PNG Image to it
            reliability_image = io.BytesIO()
            plt.savefig(reliability_image)

            # Write Image to Tensorboard
            reliability_image = tf.Summary.Image(encoded_image_string=reliability_image.getvalue(), height=7, width=7)
            summary = tf.Summary(value=[tf.Summary.Value(tag="UMAP", image=reliability_image)])
            writer.add_summary(summary)
