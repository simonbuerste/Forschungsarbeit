import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import VAE

from keras.backend.tensorflow_backend import set_session

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Add ops to save and restore all the variables.
#saver = tf.train.Saver()
imported_meta = tf.train.import_meta_graph('C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/VAE_decoder.meta')

with tf.Session() as sess:
    # Restore graph and restore weights
    imported_meta.restore(sess, tf.train.latest_checkpoint('C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/'))

    # Generating new Samples
    randoms = tf.random_normal(mean=0, stddev=1, shape=[1, VAE.n_latent])
    imgs = sess.run(VAE.decoder(randoms, 1.0))
    imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

    for img in imgs:
        plt.figure(figsize=(1, 1))
        plt.axis('off')
        plt.imshow(img, cmap='gray')
