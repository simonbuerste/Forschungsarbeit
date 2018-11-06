import tensorflow as tf
import numpy as np

import ImportDataset

from tensorflow.examples.tutorials.mnist import input_data
from keras import backend as K

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


batch_size = 64
shuffle_size = 10000
fetch_size = 64

train_data, test_data, val_data = ImportDataset.importmnist(batch_size, shuffle_size, fetch_size)

# Create One Iterator and initialize with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)    # initializer for test_data

keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

# Set Dimensions for Encoder and Decoder
dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = int(49 * dec_in_channels / 2)


# Define a Leacky ReLu Function
def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


# Defining the Encoder
def encoder(encoder_input, prob_keep):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(encoder_input, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, prob_keep)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, prob_keep)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, prob_keep)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = tf.add(mn, tf.multiply(epsilon, tf.exp(sd)), name="op_to_restore_encoder")

        return z, mn, sd


# Defining the Decoder
def decoder(sampled_z, prob_keep):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, prob_keep)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, prob_keep)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28], name="op_to_restore_decoder")
        return img


# Bringing together Encoder and Decoder
sampled, mn, sd = encoder(img, keep_prob)
dec = decoder(sampled, keep_prob)

# Computing Loss and Enforcing a Gaussian Distribution
unreshaped = tf.reshape(dec, [-1, 28 * 28])
Y_flat = tf.reshape(img, [-1, 28 * 28])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

K.set_session(tf.Session(config=config))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train the VAE
    for i in range(50):
        sess.run(train_init)
        try:
            sess.run(optimizer, feed_dict={keep_prob: 0.8})
        except tf.errors.OutOfRangeError:
            pass

        if not i % 200:
            try:
                ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd],
                                                       feed_dict={keep_prob: 1.0})
                print(i, ls, np.mean(i_ls), np.mean(d_ls))
            except tf.errors.OutOfRangeError:
                pass

    print(saver.save(sess, 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/VAE_decoder'))
