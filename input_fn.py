""" Import Data.
Import Data (i.e. Mnist) and put it into Datasets.
Set up Iterators for batching and Shuffling of Data.
"""

import tensorflow as tf


def input_fn(mode, data, params):
    """ Input Function for the Model

    Args:
        mode: (string) 'train' 'eval' or something similar. At Training, Data will be shuffled and we have multiple epochs
        data: (tf.Dataset) COntains the Data
        Params: (Parameters) contains Parameters relevant for Data Preparation (params.num_epochs, params.batch_size,..)

    """

    if mode == "train":
        buffer_size = params["buffer_size"]
    else:
        buffer_size = 1

    # Create Dataset and Iterator
    # Create Training Dataset, shuffle and batch it
    data = tf.data.Dataset.from_tensor_slices((data.images, data.labels))
    data = data.shuffle(buffer_size)    # if you want to shuffle the Data
    data = data.batch(params["batch_size"])
    data = data.prefetch(1)                     # make sure always one batch is ready to serve

    # Create initializable iterator from Data so that it can be reset at each epoch
    iterator = data.make_initializable_iterator()

    # Query the Output of the Iterator for input to the model
    img, label = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'img': img,
        'labels': label,
        'iterator_init_op': init_op
    }

    return inputs
