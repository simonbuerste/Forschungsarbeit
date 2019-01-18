""" Import Data.
Import Data (i.e. Mnist) and put it into Datasets.
Set up Iterators for batching and Shuffling of Data.
"""

import tensorflow as tf
import os


def extract_fn(tfrecord):
    # Extract features using the keys set during creation
    features = {
        'image':       tf.FixedLenFeature([], tf.string),
        'label':     tf.FixedLenFeature([], tf.int64),
        'height':    tf.FixedLenFeature([], tf.int64),
        'width':     tf.FixedLenFeature([], tf.int64),
        'depth':     tf.FixedLenFeature([], tf.int64)
    }

    # Extract the data record
    sample = tf.parse_single_example(tfrecord, features)

    # Decode image and shape from tfrecord
    img_shape = tf.stack([sample['height'], sample['width'], sample['depth']])
    image = tf.decode_raw(sample['image'], tf.uint8)
    # ensuring value range between 0 and 1
    image = tf.cast(image, tf.float32)
    image = image/255
    # reshape the image in "original Shape"
    image = tf.reshape(image, img_shape)

    # Define new Size to resize the image
    resized_size = (32, 32)
    image = tf.image.resize_images(image, resized_size)

    # Write new Size to img_shape
    sample['height'] = resized_size[0]
    sample['width'] = resized_size[1]
    img_shape = tf.stack([sample['height'], sample['width'], sample['depth']])

    label = sample['label']

    return [image, label, img_shape]


def input_fn(data_dir, mode, params):
    """ Input Function for the Model

    Args:
        mode: (string) 'train' 'test' or something similar. At Training, Data will be shuffled and we have multiple epochs
        data_dir: (directory) Contains the directory of the Data
        params: (Parameters) contains Parameters relevant for Data Preparation (params.num_epochs, params.batch_size,..)

    """

    if mode == "train":
        buffer_size = params.buffer_size
    else:
        buffer_size = 1

    # Create the link to the file
    filename = os.path.join(data_dir, mode + '.tfrecords')

    # Do pipelining explicitly on CPU
    with tf.device("/cpu:*"):
        # Pipeline of dataset and iterator
        dataset = tf.data.TFRecordDataset([filename])
        dataset = dataset.map(extract_fn, num_parallel_calls=4)
        # dataset = dataset.map(augmentation_fn, num_parallel_calls=2)
        # Create Training Dataset, shuffle and batch it
        dataset = dataset.shuffle(buffer_size)    # if you want to shuffle the Data
        dataset = dataset.batch(params.batch_size)
        dataset = dataset.prefetch(4)                     # make sure always one batch is ready to serve

        # Create initializable iterator from Data so that it can be reset at each epoch
        iterator = dataset.make_initializable_iterator()

        # Query the Output of the Iterator for input to the model
        img, label, img_shape = iterator.get_next()
        init_op = iterator.initializer

        # Build and return a dictionary containing the nodes / ops
        inputs = {
            'img': img,
            'labels': label,
            'img_shape': img_shape,
            'iterator_init_op': init_op
        }

    return inputs
