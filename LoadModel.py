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
saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore graph and restore weights
    saver.restore(sess, 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/VAE_decoder')

    # Generating new Samples
    randoms = [np.random.normal(0, 1, VAE.n_latent) for _ in range(2)]
    imgs = sess.run(VAE.decoder(randoms, 1.0))
    imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

    for img in imgs:
        plt.figure(figsize=(1, 1))
        plt.axis('off')
        plt.imshow(img, cmap='gray')
