from typing import Any, Union

import tensorflow as tf


# Define a Leacky ReLu Function
def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


# Defining the Encoder
def encoder(encoder_input, prob_keep, is_training, params):
    x = encoder_input
    latent_list = []
    with tf.name_scope('original'):
        tf.summary.image('original', x, max_outputs=6)
    print('-------Encoder-------')
    for k in range(4):
        print(x.get_shape())
        x = tf.layers.conv2d(x, filters=16*(2**k), kernel_size=3, strides=1, padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x,alpha=0.2)

        if k<3:
            x = tf.layers.max_pooling2d(x,2,2)
            latent_list.append(x)

        # Last layer average pooling
    x = tf.layers.average_pooling2d(x,4,4)
    print(x.get_shape())
    x = tf.contrib.layers.flatten(x)
    print(x.get_shape())

    z = tf.layers.dense(x,units=params.n_latent,kernel_initializer=tf.contrib.layers.xavier_initializer())
    #z = tf.divide(z,tf.norm(z)+1e-10) # Normalizing l2 norm
    print(z.get_shape())
    print('-------Encoder-------')

    return z, latent_list


# Defining the Decoder
def decoder(sampled_z, prob_keep, is_training,params):
    print('-------Decoder-------')
    print(sampled_z.get_shape())
    latent_list= []
    reshaped_dim = [-1, 2, 2, 128]
    inputs_decoder = int(2*2*128)
    x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu,kernel_initializer=tf.contrib.layers.xavier_initializer())
    print(x.get_shape())
    x = tf.reshape(x, reshaped_dim)
    print(x.get_shape())
    for k in range(4):
        x = tf.layers.conv2d_transpose(x,filters=max(16,16*(2**(3-k-1))),kernel_size=4, strides=2, padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x,training=is_training)
        x = tf.nn.leaky_relu(x,alpha=0.2)
        print(x.get_shape())
        latent_list.append(x)

    #remove last layer from list
    latent_list = latent_list[:3]
    reconstructed_mean = tf.layers.conv2d(x,filters=params.channels,kernel_size=3,padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer())#(activation=tf.nn.sigmoid)tf.reshape(x, shape=[-1, params.resize_height, params.resize_width, params.channels])
    print(reconstructed_mean.get_shape())
    print('-------Decoder-------')
    with tf.name_scope('reconstructed'):
        tf.summary.image('reconstructed', tf.nn.sigmoid(reconstructed_mean), max_outputs=6)
    return reconstructed_mean, latent_list


def build_model(inputs, keep_prob,is_training, params):

    original_img = inputs["img"]
    # Bringing together Encoder and Decoder
    sampled, latent_list_enc = encoder(original_img, keep_prob,is_training, params)
    reconstructed_mean, latent_list_dec = decoder(sampled, keep_prob,is_training, params)

    loss_square = tf.losses.mean_squared_error(labels=original_img,predictions=tf.sigmoid(reconstructed_mean))

    return loss_square, sampled


def ae_model_fn(mode, inputs, params, reuse=False):
    """Model ae function defining the graph operations

    Args:
        mode:   (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels,...)
        params: (dict) contains hyperparameters of the model (i.e. learning_rate,...)
        reuse:  (bool) whether to reuse Variables (weights, bias,...)

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """

    if mode == 'train':
        p_dropout = params.p_dropout
        is_training = True
    else:
        p_dropout = 1
        is_training = False
# -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('vae_model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        loss_likelihood, sampled = build_model(inputs, p_dropout,is_training, params)

    # Define the Loss
    loss = tf.reduce_mean(loss_likelihood)

    # Define training step that minimizes the loss with the Adam optimizer
    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(0.0005)
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("vae_metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
            'neg_log_likelihodd': tf.metrics.mean(loss_likelihood)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="vae_metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('neg_log_likelihood', loss_likelihood)

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['loss'] = loss
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if mode == 'train':
        model_spec['train_op'] = train_op
    elif mode == 'cluster':
        model_spec['sample'] = sampled

    return model_spec
