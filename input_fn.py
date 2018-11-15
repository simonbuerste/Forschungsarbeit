""" Import Data.
Import Data (i.e. Mnist) and put it into Datasets.
Set up Iterators for batching and Shuffling of Data.
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def importmnist(batch_size, shuffle_size, fetch_size):
    mnist = input_data.read_data_sets('/MNIST_data')

    # Defining the Input Data
    train, val, test = mnist.train, mnist.validation, mnist.test

    # Create Dataset and Iterator
    # Create Training Dataset, shuffle and batch it
    train_data = tf.data.Dataset.from_tensor_slices((train.images, train.labels))
    train_data = train_data.shuffle(shuffle_size)  # if you want to shuffle the Data
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(buffer_size=batch_size)

    # Create Test Dataset
    test_data = tf.data.Dataset.from_tensor_slices((test.images, test.labels))
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(buffer_size=fetch_size)

    # Create Validation Dataset
    val_data = tf.data.Dataset.from_tensor_slices((val.images, val.labels))
    val_data = val_data.batch(batch_size)
    val_data = val_data.prefetch(buffer_size=fetch_size)

    return train_data, test_data, val_data

def input_fn(mode, data, params):
    """ Input Function for the Model

    Args:
        mode: (string) 'train' 'eval' or something similar. At Training, Data will be shuffled and we have multiple epochs
        data: (tf.Dataset) COntains the Data
        Params: (Parameters) contains Parameters relevant for Data Preparation (params.num_epochs, params.batch_size,..)

    """
    
    if (mode == "train"):
        buffer_size = params.buffer_size
    else:
        buffer_size = 1


