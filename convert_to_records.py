""" Convert MNIST data to TFRecords file format with Example protos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, save_dir, name):
    """Converts a dataset to tfrecords"""
    images = data_set.images
    labels = data_set.labels
    num_samples = data_set.num_examples

    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(save_dir, name + '.tfrecords')
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_samples):
            image_raw = images[index].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height':   _int64_feature(rows),
                        'width':    _int64_feature(cols),
                        'depth':    _int64_feature(depth),
                        'label':    _int64_feature(int(labels[index])),
                        'image':    _bytes_feature(image_raw)
                    }))
            writer.write(example.SerializeToString())


if __name__ == '__main__':

    data_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Data/F-MNIST'
    # data_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'MNIST_data')

    # Get the data
    dataset = mnist.read_data_sets(data_dir,
                                   dtype=tf.uint8,
                                   reshape=False)

    # Convert and write result to TFRecords
    convert_to(dataset.train, data_dir, 'train')
    #convert_to(dataset.validation, data_dir, 'validation')
    convert_to(dataset.test, data_dir, 'test')
