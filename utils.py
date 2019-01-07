
"""General utility functions"""

import json
import logging
import os
import numpy as np
import tensorflow as tf
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


def samples_latentspace(model_spec):
    sampled = model_spec['sample']
    return sampled


def visualize_embeddings(sess, log_dir, writer, params):

    sub_latentspace = []
    sub_metadata = []
    for i in range(params.num_epochs):
        metadata = os.path.join(log_dir, ('metadata' + str(i + 1) + '.tsv'))
        img_latentspace = os.path.join(log_dir, ('latentspace' + str(i + 1) + '.txt'))

        latentspace = np.loadtxt(img_latentspace)
        features = tf.Variable(latentspace, name=('latentspace' + str(i+1)))
        sub_latentspace.append(features)
        sub_metadata.append(metadata)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, log_dir)

    config = projector.ProjectorConfig()
    for i in range(params.num_epochs):
        embedding = config.embeddings.add()
        embedding.tensor_name = sub_latentspace[i].name
        embedding.metadata_path = sub_metadata[i]

    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(writer, config)
