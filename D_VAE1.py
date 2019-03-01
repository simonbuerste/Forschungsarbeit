import tensorflow as tf


# Define a Leacky ReLu Function
def lrelu(x, alpha=0.2):
    return tf.maximum(x, tf.multiply(x, alpha))


def selfattentionlayer(x, iteration, sigma):

    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h*w
    downsampled_num = location_num//4

    reduced_channel = num_channels//8

    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1

    w_f = tf.get_variable(name='w_f_%d' % iteration, initializer=tf.contrib.layers.xavier_initializer(), shape=[k_h, k_w, x.get_shape()[-1], reduced_channel])
    w_g = tf.get_variable(name='w_g_%d' % iteration, initializer=tf.contrib.layers.xavier_initializer(), shape=[k_h, k_w, x.get_shape()[-1], reduced_channel])
    w_h = tf.get_variable(name='w_h_%d' % iteration, initializer=tf.contrib.layers.xavier_initializer(), shape=[k_h, k_w, x.get_shape()[-1], num_channels])
    w_attn = tf.get_variable(name='w_attn_%d' % iteration, initializer=tf.contrib.layers.xavier_initializer(), shape=[k_h, k_w, x.get_shape()[-1], num_channels])

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

    # sigma = tf.get_variable('sigma_ratio_%d' % iteration, [], initializer=tf.constant_initializer(1.0))
    output = tf.nn.conv2d(attn_g, w_attn, strides=[1, d_h, d_w, 1], padding='SAME')
    return x + sigma*output


# Defining the Encoder
def encoder(encoder_input, is_training, params):
    x = tf.reshape(encoder_input, shape=[-1, params.resize_height, params.resize_width, params.channels])
    print('-------Encoder-------')
    sigma = tf.placeholder(tf.float32, [], name="sigma_ratio")
    for k in range(4):
        print(x.get_shape())
        x = tf.layers.conv2d(x, filters=16*(2**k), kernel_size=3, strides=1, padding='same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        if k < 3:
            x = tf.layers.max_pooling2d(x, 2, 2)
        #if k == 3:
        x = selfattentionlayer(x, k, sigma)

    # Last layer average pooling
    x = tf.layers.average_pooling2d(x, 4, 4)
    print(x.get_shape())
    x = tf.contrib.layers.flatten(x)
    print(x.get_shape())

    z_mu = tf.layers.dense(x, units=params.n_latent, kernel_initializer=tf.contrib.layers.xavier_initializer())
    z_log_sigma_sq = tf.layers.dense(x, units=params.n_latent, kernel_initializer=tf.contrib.layers.xavier_initializer())
    print(z_mu.get_shape())

    q_z = tf.distributions.Normal(loc=z_mu, scale=tf.sqrt(tf.exp(z_log_sigma_sq)))
    z = q_z.sample()
    print('-------Encoder-------')

    return z, z_mu, z_log_sigma_sq, q_z


# Defining the Decoder
def decoder(sampled_z, is_training, params):
    print('-------Decoder-------')
    print(sampled_z.get_shape())

    reshaped_dim = [-1, 2, 2, params.n_latent]
    inputs_decoder = int(2*2*params.n_latent)
    x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
    print(x.get_shape())
    x = tf.reshape(x, reshaped_dim)
    print(x.get_shape())
    for k in range(4):
        x = tf.layers.conv2d_transpose(x, filters=max(16, 16*(2**(3-k-1))), kernel_size=4, strides=2, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        #if k == 1:
        #    x = selfattentionlayer(x, k)
        print(x.get_shape())

    reconstructed_mean = tf.layers.conv2d(x, filters=params.channels, kernel_size=3, padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer()) #(activation=tf.nn.sigmoid)tf.reshape(x, shape=[-1, params.resize_height, params.resize_width, params.channels])
    
    #reconstructed_mean = tf.layers.conv2d_transpose(x, filters=params.channels, kernel_size=4, strides=2, padding='same',
    #                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    print(reconstructed_mean.get_shape())
    print('-------Decoder-------')
    return reconstructed_mean


def build_model(inputs, is_training, params):

    original_img = inputs["img"]
    # Bringing together Encoder and Decoder
    sampled, z_mu, z_log_sigma_sq, q_z, sigma_placeholder = encoder(original_img, is_training, params)
    reconstructed_mean = decoder(sampled, is_training, params)

    # Calculate log likelihood
    #loss_likelihood = original_img * tf.log(1e-10+ reconstructed_mean) + (1- original_img)*tf.log(1e-10+1-reconstructed_mean)
    #loss_likelihood = -tf.reduce_mean(tf.reduce_sum(loss_likelihood,[1,2,3]),0)

    # Another way for the loss
    #help_p = tf.distributions.Bernoulli(logits=reconstructed_mean)
    temperature = tf.constant(1.0)
    help_p = tf.distributions.RelaxedOneHotCategorical(temperature, logits=reconstructed_mean)
    loss_likelihood = -tf.reduce_mean(tf.reduce_sum(help_p.log_prob(original_img), [1, 2, 3]))

    # Calculate KL loss
    p_z = tf.distributions.Normal(tf.zeros_like(sampled), tf.ones_like(sampled))
    kl_loss = q_z.kl_divergence(p_z)
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))

    return loss_likelihood, kl_loss, sampled, reconstructed_mean, sigma_placeholder


def vae_model_fn(mode, inputs, params, reuse=False):
    """Model vae function defining the graph operations

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
    with tf.variable_scope('vae_model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        loss_likelihood, kl_loss, sampled, reconstructed_mean, sigma_placeholder = build_model(inputs, is_training, params)

    # Define the Loss
    loss = tf.reduce_mean(loss_likelihood + kl_loss)

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
    with tf.variable_scope("vae_metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
            'kl_loss': tf.metrics.mean(kl_loss),
            'neg_log_likelihood': tf.metrics.mean(loss_likelihood)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="vae_metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('kl_loss', kl_loss)
    tf.summary.scalar('neg_log_likelihood', loss_likelihood)
    # Summary for reconstruction and original image with max_outpus images
    tf.summary.image('Original Image', inputs['img'], max_outputs=6, collections=None, family=None)
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

    if mode == 'train':
        model_spec['train_op'] = train_op
    elif mode == 'cluster':
        model_spec['sample'] = sampled

    return model_spec
