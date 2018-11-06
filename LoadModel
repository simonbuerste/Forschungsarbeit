import tensorflow as tf
import numpy as np
import VAE
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from keras.backend.tensorflow_backend import set_session

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore graph and restore weights
    saver.restore(sess, 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/VAE_decoder')

    # Now, access the op that you want to run.
    graph = tf.get_default_graph()
    op_decoder = graph.get_tensor_by_name("op_to_restore_decoder:0")

    # Generating new Samples
    randoms = [np.random.normal(0, 1, VAE.n_latent) for _ in range(2)]
    imgs = sess.run(op_decoder, feed_dict={VAE.sampled: randoms, VAE.keep_prob: 1.0})
    imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

    for img in imgs:
        plt.figure(figsize=(1, 1))
        plt.axis('off')
        plt.imshow(img, cmap='gray')
