""" Import Data.
Import Data (i.e. Mnist) and put it into Datasets.
Set up Iterators for batching and Shuffling of Data.
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/home/s1279/no_backup/s1279/MNIST_data')

batch_size = 64
shuffle_size = 10000  # Set shuffle_size to len(Input) if no shuffling is required

# Defining the Input Data
train, val, test = mnist.train, mnist.validation, mnist.test

# Create Dataset and Iterator
# Create Training Dataset, shuffle and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(shuffle_size)  # if you want to shuffle the Data
train_data = train_data.batch(batch_size)

# Create Test Dataset
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

# Create Validation Dataset
val_data = tf.data.Dataset.from_tensor_slices(val)
val_data = val_data.batch(batch_size)

# Create One Iterator and initialize with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)    # initializer for test_data
val_init = iterator.make_initializer(val_data)
