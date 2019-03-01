""" Convert datasets to TFRecords file format with Example protos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tarfile
import random
import numpy as np
from six.moves import cPickle as pickle
from six.moves import xrange

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

# CIFAR-10/CIFAR-100 Data
CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'


def download_and_extract(data_dir):
    # download CIFAR-10 if not already downloaded.
    tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                  CIFAR_DOWNLOAD_URL)
    tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
               'r:gz').extractall(data_dir)


def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 6)]
    file_names['test'] = ['test_batch']

    return file_names


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def convert_dataset_to_tfrecord(data_set, save_dir, name):
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


def read_pickle_from_file(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding='bytes')
        else:
            data_dict = pickle.load(f)
        return data_dict


def convert_pickle_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as writer:
        # draw 10 random number for getting 10 random classes from Imagenet (fixed value for reproducibility)
        #class_id = [145, 153, 289, 404, 405, 510, 805, 817, 867, 950]  # random.sample(range(0, 999), 10)
        class_id = [153, 156, 161, 174, 197, 207, 215, 216, 218, 224, 227, 230, 236, 254, 260]  # 15 dog classes (also used in DAC)

        count = np.zeros(shape=len(class_id))
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict['data']
            mean_img = data_dict['mean']
            labels = data_dict['labels']
            # Labels are indexed from 1, shift it so that indexes start at 0 (imagenet)
            labels = [i - 1 for i in labels]

            num_entries_in_batch = len(labels)

            for i in range(num_entries_in_batch):
                if labels[i] in class_id:
                    labels[i] = class_id.index(labels[i])  # put the labels into the range of 0 to no. clusters
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'height':   _int64_feature(64),
                                'width':    _int64_feature(64),
                                'depth':    _int64_feature(3),
                                'image':    _bytes_feature(data[i].tobytes()),
                                'mean_img': _bytes_feature(mean_img.tobytes()),
                                'label':    _int64_feature(labels[i])
                            }))
                    writer.write(example.SerializeToString())
                    count[labels[i]] += 1  # count number of samples per class
        for idx, num in enumerate(count):
            print('Number of samples of class %d: %d' % (idx, num))
        print('Total Number of samples %d' % np.sum(count))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="imagenet",
                        help="Dataset which should be converted")
    data_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Data'
    # data_dir = os.path.join(os.path.expanduser('~'), 'no_backup', 's1279', 'MNIST_data')
    parser.add_argument('--data_dir', default=data_dir,
                        help="Directory containing the dataset")

    # Load the parameters
    args = parser.parse_args()
    data_dir = os.path.join(data_dir, args.dataset)

    if args.dataset == "mnist" or args.dataset == "f-mnist":  # Mnist / Fashion Mnist data
        dataset = mnist.read_data_sets(data_dir,
                                       dtype=tf.uint8,
                                       reshape=False)

        # Convert and write datasets to TFRecords
        convert_dataset_to_tfrecord(dataset.train, data_dir, 'train')
        # convert_to(dataset.validation, data_dir, 'validation')
        convert_dataset_to_tfrecord(dataset.test, data_dir, 'test')
    elif args.dataset == "cifar-10":  # CIFAR-10
        # Convert and write pickle data to tfrecord
        download_and_extract(data_dir)
        file_names = _get_file_names()
        input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
        for mode, files in file_names.items():
            input_files = [os.path.join(input_dir, f) for f in files]
            output_file = os.path.join(data_dir, mode + '.tfrecords')
            try:
                os.remove(output_file)
            except OSError:
                pass
            convert_pickle_to_tfrecord(input_files, output_file)
    elif args.dataset == "cifar-100":  # CIFAR-100
        # Convert and write pickle data to tfrecord
        download_and_extract(data_dir)
        file_names = {}
        file_names['train'] = ['train']
        file_names['test'] = ['test']
        input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
        for mode, files in file_names.items():
            input_files = [os.path.join(input_dir, f) for f in files]
            output_file = os.path.join(data_dir, mode + '.tfrecords')
            try:
                os.remove(output_file)
            except OSError:
                pass
            convert_pickle_to_tfrecord(input_files, output_file)
    elif args.dataset == "imagenet":
        file_names = {}
        file_names['train'] = ['train_data_batch_%d' % i for i in xrange(1, 11)]
        #file_names['test'] = ['val_data']

        for mode, files in file_names.items():
            input_files = [os.path.join(data_dir, f) for f in files]
            output_file = os.path.join(data_dir, mode + '.tfrecords')
            try:
                os.remove(output_file)
            except OSError:
                pass
            convert_pickle_to_tfrecord(input_files, output_file)
    else:
        print('Unknown/not supported dataset')
