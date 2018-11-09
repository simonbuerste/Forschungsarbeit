import tensorflow as tf

# Define Variable for keep_Probability (Value will change depending on Session)
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
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
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
        z = tf.add(mn, tf.multiply(epsilon, tf.exp(sd)))

        return z, mn, sd


# Defining the Decoder
def decoder(sampled_z, prob_keep):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
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
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img


def run_vae(img):
    # Bringing together Encoder and Decoder
    sampled, mn, sd = encoder(img, keep_prob)
    dec = decoder(sampled, keep_prob)
    # Computing Loss and Enforcing a Gaussian Distribution
    with tf.variable_scope("optimization", reuse=tf.AUTO_REUSE):
        unreshaped = tf.reshape(dec, [-1, 28 * 28])
        y_flat = tf.reshape(img, [-1, 28 * 28])
        img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, y_flat), 1)
        latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
        loss = tf.reduce_mean(img_loss + latent_loss)
        optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
        return optimizer, loss
