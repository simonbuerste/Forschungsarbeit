""" Import Data.
Import Data (i.e. Mnist) and put it into Datasets.
Set up Iterators for batching and Shuffling of Data.
"""

import tensorflow as tf
import os


def extract_fn(tfrecord, params):
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
    img_shape = tf.stack([sample['depth'], sample['height'], sample['width']])
    image = tf.decode_raw(sample['image'], tf.uint8)
    # ensuring value range between 0 and 1
    image = tf.cast(image, tf.float32)
    image = image/255
    # reshape the image in "original Shape"
    image = tf.reshape(image, img_shape)
    # Transpose Image for tensorflow notation (heigth, width, num_channel)
    image = tf.transpose(image, (1, 2, 0))
    # Define new Size to resize the image
    image = tf.image.resize_images(image, (params.resize_height, params.resize_width))

    # Write new Size to img_shape
    sample['height'] = params.resize_height
    sample['width'] = params.resize_width
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
        batch_size = params.train_batch_size
    else:
        buffer_size = 1
        batch_size = params.eval_batch_size

    # Create the link to the file
    filename = os.path.join(data_dir, mode + '.tfrecords')

    # Write channel number to params dict
    if "MNIST" in filename:
        params.channels = 1
    elif "CIFAR-10" or "IMAGENET" in filename:
        params.channels = 3


    # Write class number to params dict
    if "MNIST" in filename:  # Mnist and Fashion Mnist
        params.channels = 1
        params.num_classes = 10
    elif "CIFAR-10" or "IMAGENET" in filename:
        params.channels = 3
        if "CIFAR-100" in filename:
            params.num_classes = 20
        elif "IMAGENET-10" in filename:
            params.num_classes = 10
        else:  # CIFAR-10 case
            params.num_classes = 10

    # Do pipelining explicitly on CPU
    with tf.device("/cpu:*"):
        # Pipeline of dataset and iterator
        dataset = tf.data.TFRecordDataset([filename])
        dataset = dataset.map(lambda x: extract_fn(x, params), num_parallel_calls=2)
        # dataset = dataset.map(augmentation_fn, num_parallel_calls=2)
        # Create Training Dataset, shuffle and batch it
        dataset = dataset.shuffle(buffer_size)    # if you want to shuffle the Data
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)                     # make sure always one batch is ready to serve

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
