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

restore_dir = 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/'

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    # Get Direction of saved model
    best_save_path = os.path.join(restore_dir, 'best_weights/', 'after-epoch')
    restore_from = tf.train.latest_checkpoint(best_save_path)
    # Restore graph and restore weights
    saver.restore(sess, restore_from)

    # Generating new Samples for Clustering Task with Encoder


    # Generating new Pictures with decoder
    # randoms = tf.random_normal(mean=0, stddev=1, shape=[1, VAE.n_latent])
    # imgs = sess.run(VAE.decoder(randoms, 1.0))
    # imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
    #
    # for img in imgs:
    #     plt.figure(figsize=(1, 1))
    #     plt.axis('off')
    #     plt.imshow(img, cmap='gray')
