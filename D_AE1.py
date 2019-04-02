import tensorflow as tf
import numpy as np


# Define a Leacky ReLu Function
def lrelu(x, alpha=0.2):
    return tf.maximum(x, tf.multiply(x, alpha))


def selfattentionlayer(x, name_scope, sigma):

    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h*w
    downsampled_num = location_num//4
    reduced_channel = np.maximum(num_channels//8, 1)

    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1

    for i in range(3):
        w_f = tf.get_variable(name='w_f_' + name_scope + '_%d' % i, initializer=tf.contrib.layers.xavier_initializer(), shape=[k_h, k_w, x.get_shape()[-1], reduced_channel])
        w_g = tf.get_variable(name='w_g_' + name_scope + '_%d' % i, initializer=tf.contrib.layers.xavier_initializer(), shape=[k_h, k_w, x.get_shape()[-1], reduced_channel])
        w_h = tf.get_variable(name='w_h_' + name_scope + '_%d' % i, initializer=tf.contrib.layers.xavier_initializer(), shape=[k_h, k_w, x.get_shape()[-1], num_channels])
        w_attn = tf.get_variable(name='w_attn_' + name_scope + '_%d' % i, initializer=tf.contrib.layers.xavier_initializer(), shape=[k_h, k_w, x.get_shape()[-1], num_channels])

        theta = tf.nn.conv2d(x, w_f, strides=[1, d_h, d_w, 1], padding='SAME')
        phi = tf.nn.conv2d(x, w_g, strides=[1, d_h, d_w, 1], padding='SAME')
        g = tf.nn.conv2d(x, w_h, strides=[1, d_h, d_w, 1], padding='SAME')

        phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
        g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)

        theta = tf.reshape(theta, [-1, location_num, reduced_channel])
        phi = tf.reshape(phi, [-1, downsampled_num, reduced_channel])
        g = tf.reshape(g, [-1, downsampled_num, num_channels])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [-1, h, w, num_channels])
        if i == 0:
            output = tf.nn.conv2d(attn_g, w_attn, strides=[1, d_h, d_w, 1], padding='SAME')
        else:
            output += tf.nn.conv2d(attn_g, w_attn, strides=[1, d_h, d_w, 1], padding='SAME')

    return x + sigma*output


# Defining the Encoder
def encoder(encoder_input, is_training, params, sigma):
    x = tf.reshape(encoder_input, shape=[-1, params.resize_height, params.resize_width, params.channels])
    print('-------Encoder-------')
    for k in range(4):
        print(x.get_shape())
        x = tf.layers.conv2d(x, filters=params.filter_first_layer*(2**k), kernel_size=4, strides=1, padding='same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        #if k < 3:
        x = tf.layers.max_pooling2d(x, 2, 2)

    # Last layer average pooling
    #x = tf.layers.average_pooling2d(x, 4, 4)
    print(x.get_shape())
    x = tf.contrib.layers.flatten(x)
    print(x.get_shape())

    z = tf.layers.dense(x, units=params.n_latent, kernel_initializer=tf.contrib.layers.xavier_initializer())
    #z = tf.divide(z,tf.norm(z)+1e-10) # Normalizing l2 norm
    print(z.get_shape())
    print('-------Encoder-------')

    return z


# Defining the Decoder
def decoder(sampled_z, is_training, params, sigma):
    print('-------Decoder-------')
    print(sampled_z.get_shape())

    reshaped_dim = [-1, 2, 2, params.filter_first_layer*(2**3)]
    inputs_decoder = int(2*2*params.filter_first_layer*(2**3))
    #x = selfattentionlayer(sampled_z, 'decoder_0', sigma)
    #x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
    print(x.get_shape())
    x = tf.reshape(x, reshaped_dim)
    print(x.get_shape())

    for k in range(3):
        # if k == 0:
        #     x = selfattentionlayer(x, 'decoder_%d' % (k+1), sigma)
        #     x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.layers.conv2d_transpose(x, filters=max(params.filter_first_layer, params.filter_first_layer*(2**(3-k-1))), kernel_size=4, strides=2, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print(x.get_shape())

    reconstructed_mean = tf.layers.conv2d_transpose(x, filters=params.channels, kernel_size=3, strides=2, padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
    #reconstructed_mean = tf.layers.conv2d_transpose(x, filters=params.channels, kernel_size=4, strides=2, padding='same',
    #                                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    print(reconstructed_mean.get_shape())
    print('-------Decoder-------')
    return reconstructed_mean


def build_model(inputs, is_training, params):

    original_img = inputs["img"]
    sigma = tf.placeholder(tf.float32, [], name="sigma_ratio")
    # Bringing together Encoder and Decoder
    sampled = encoder(original_img, is_training, params, sigma)
    reconstructed_mean = decoder(sampled, is_training, params, sigma)

    dyn_rage = tf.reduce_max(original_img) - tf.reduce_min(original_img)
    ssim_loss = tf.reduce_mean(1.0 - (tf.image.ssim(original_img, tf.sigmoid(reconstructed_mean), max_val=dyn_rage)+1.0)/2.0)  # bring it to 0-1 format

    #loss_square = - tf.reduce_mean(original_img*tf.log(tf.sigmoid(reconstructed_mean)+1e-20) + (1-original_img)*tf.log(1-tf.sigmoid(reconstructed_mean)+1e-20))
    loss_square = tf.losses.mean_squared_error(labels=original_img, predictions=tf.sigmoid(reconstructed_mean))
    #loss_square = tf.norm(original_img-tf.sigmoid(reconstructed_mean))
    #loss_square = ssim_loss

    return loss_square, sampled, reconstructed_mean, sigma


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
        is_training = True
    else:
        is_training = False
# -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('ae_model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        loss_likelihood, sampled, reconstructed_mean, sigma_placeholder = build_model(inputs, is_training, params)

    # Define the Loss
    loss = tf.reduce_mean(loss_likelihood)

    # Define training step that minimizes the loss with the Adam optimizer
    learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate")
    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate_ph)
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("ae_metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
            'neg_log_likelihood': tf.metrics.mean(loss_likelihood)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="ae_metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('neg_log_likelihood', loss_likelihood)
    # Summary for reconstruction and original image with max_outpus images
    tf.summary.image('Original Image', inputs['img'], max_outputs=6, collections=None, family=None)
    latent_img = tf.reshape(sampled, [-1, 1, params.n_latent, 1])
    tf.summary.image('Latent Space', latent_img, max_outputs=6, collections=None, family=None)
    tf.summary.image('Reconstructions', tf.sigmoid(reconstructed_mean), max_outputs=6, collections=None, family=None)

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
    model_spec['reconstructions'] = tf.sigmoid(reconstructed_mean)
    model_spec['sigma_placeholder'] = sigma_placeholder
    model_spec['learning_rate_placeholder'] = learning_rate_ph
    model_spec['gamma_placeholder'] = tf.placeholder(tf.float32, [], name="gamma")
    model_spec['lambda_r_placeholder'] = tf.placeholder(tf.float32, shape=[], name='Reconstruction_regularization')
    model_spec['lambda_c_placeholder'] = tf.placeholder(tf.float32, shape=[], name='center_sim_regularization')
    model_spec['lambda_d_placeholder'] = tf.placeholder(tf.float32, shape=[], name='discriminative_regularization')
    model_spec['lambda_b_placeholder'] = tf.placeholder(tf.float32, shape=[], name='inter_cluster_sim_regularization')
    model_spec['lambda_w_placeholder'] = tf.placeholder(tf.float32, shape=[], name='intra_cluster_sim_regularization')

    if mode == 'train':
        model_spec['train_op'] = train_op
    elif mode == 'cluster':
        model_spec['sample'] = sampled

    return model_spec
