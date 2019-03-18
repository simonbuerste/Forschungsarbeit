import tensorflow as tf
import numpy as np


# Define a Leacky ReLu Function
def lrelu(x, alpha=0.2):
    return tf.maximum(x, tf.multiply(x, alpha))


# Defining the Encoder
def encoder(encoder_input, is_training, params, sigma):
    x = tf.reshape(encoder_input, shape=[-1, params.resize_height, params.resize_width, params.channels])
    print('-------Encoder-------')
    for k in range(4):
        print(x.get_shape())
        x = tf.layers.conv2d(x, filters=16*(2**k), kernel_size=4, strides=1, padding='same',
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

    reshaped_dim = [-1, 2, 2, 16*(2**3)]
    inputs_decoder = int(2*2*16*(2**3))
    #x = selfattentionlayer(sampled_z, 'decoder_0', sigma)
    #x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
    print(x.get_shape())
    x = tf.reshape(x, reshaped_dim)
    print(x.get_shape())

    for k in range(3):
        #x = selfattentionlayer(x, 'decoder_%d' % (k+1), sigma)
        #x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.layers.conv2d_transpose(x, filters=max(16, 16*(2**(3-k-1))), kernel_size=4, strides=2, padding='same',
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

    loss_square = tf.losses.mean_squared_error(labels=original_img, predictions=tf.sigmoid(reconstructed_mean))
    #loss_square = tf.norm(original_img - tf.sigmoid(reconstructed_mean))

    return loss_square, sampled, reconstructed_mean, sigma


def fsae_model_fn(mode, inputs, params, reuse=False):
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
    with tf.variable_scope('fsae_model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        loss_likelihood, sampled, reconstructed_mean, sigma_placeholder = build_model(inputs, is_training, params)
    with tf.variable_scope('feature_selection', reuse=reuse):
        m = params.n_latent//2
        help_p = tf.get_variable(name='temporary_p', shape=(params.n_latent, 1), initializer=tf.zeros_initializer)
        y_t = tf.get_variable(name='y_t', shape=params.n_latent, initializer=tf.zeros_initializer)
        P = tf.get_variable(name='projection_matrix', shape=(params.n_latent, m), initializer=tf.initializers.identity)
        gamma = tf.get_variable(name='gamma', shape=(), initializer=tf.zeros_initializer)

    z = inputs["img"]/tf.norm(inputs["img"], ord=1)
    z_flat = tf.contrib.layers.flatten(z)
    r = tf.reshape(tf.reduce_sum(z_flat*z_flat, 1), [-1, 1])
    sim_original_img = r - 2*tf.matmul(z_flat, z_flat, transpose_b=True) + tf.transpose(r)
    sum_anchor = tf.constant(15)

    t = tf.constant(1.0)

    inter_cluster_val, inter_cluster_idx = tf.nn.top_k(sim_original_img, k=sum_anchor)
    min_inter_cluster_val = tf.reduce_min(inter_cluster_val, axis=1)
    mask_less = tf.less(sim_original_img, min_inter_cluster_val)
    sim_inter_cluster = sim_original_img*(tf.cast(mask_less, sim_original_img.dtype))
    S_w = tf.exp(-sim_inter_cluster/t)
    L_w = tf.diag_part(S_w) - S_w

    betweeen_cluster_val, between_cluster_idx = tf.nn.top_k(-sim_original_img, k=sum_anchor)
    max_betweeen_cluster_val = tf.reduce_max(-betweeen_cluster_val, axis=1)
    mask_greater = tf.greater(sim_original_img, max_betweeen_cluster_val)
    sim_between_cluster = sim_original_img*(tf.cast(mask_greater, sim_original_img.dtype))
    S_b = tf.exp(-sim_between_cluster/t)
    L_b = tf.diag_part(S_b) - S_b

    # Optimization of trace-ratio problem
    for i in range(params.n_latent):
        help_p.assign(tf.zeros(help_p.get_shape()))
        help_p[i].assign(tf.constant(1.0))
        feature = tf.matmul(sampled, help_p)
        y_t[i].assign(tf.trace(tf.matmul(tf.matmul(feature, L_w, transpose_a=True), feature)))/(tf.trace(tf.matmul(tf.matmul(feature, L_b, transpose_a=True), feature)))

    _, leading_features = tf.nn.top_k(-y_t, k=m)

    for i in range(m):
        P.assign(tf.zeros(P.get_shape()))
        P[leading_features[i], i].assign(tf.constant(1.0))

    new_sample = tf.matmul(sampled, P)
    gamma.assign(tf.trace(tf.matmul(tf.matmul(new_sample, L_w, transpose_a=True), new_sample)))/(tf.trace(tf.matmul(tf.matmul(new_sample, L_b, transpose_a=True), new_sample)))
    # Define the Loss
    lambd = tf.placeholder(tf.float32, [], name="lambda")
    trace_loss_feature = tf.trace(tf.matmul(tf.matmul(new_sample, (L_w-gamma*L_b), transpose_a=True), new_sample))
    loss = tf.reduce_mean(0.5*loss_likelihood + 0.5*lambd*trace_loss_feature)

    # Define training step that minimizes the loss with the Adam optimizer
    learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate")
    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate_ph)
        global_step = tf.train.get_or_create_global_step()
        vars_ae_train_op = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fsae_model")
        vars_trace_ratio_train_op = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "feature_selection")
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_ae_train_op)
            train_op_trace_ratio = optimizer.minimize(loss, global_step=global_step, var_list=vars_trace_ratio_train_op)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("fsae_metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
            'neg_log_likelihood': tf.metrics.mean(loss_likelihood)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="fsae_metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('neg_log_likelihood', loss_likelihood)
    # Summary for reconstruction and original image with max_outpus images
    tf.summary.image('Original Image', inputs['img'], max_outputs=6, collections=None, family=None)
    latent_img = tf.reshape(sampled, [-1, 1, params.n_latent, 1])
    tf.summary.image('Latent Space', latent_img, max_outputs=6, collections=None, family=None)
    latent_img_feature_selected = tf.reshape(new_sample, [-1, 1, m, 1])
    tf.summary.image('Latent Space Feature Selected', latent_img_feature_selected, max_outputs=6, collections=None, family=None)
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
    model_spec['lambda_r_placeholder'] = lambd
    model_spec['lambda_c_placeholder'] = tf.placeholder(tf.float32, [], name="lambda_c")
    model_spec['lambda_d_placeholder'] = tf.placeholder(tf.float32, [], name="lambda_d")
    model_spec['lambda_b_placeholder'] = tf.placeholder(tf.float32, [], name="lambda_b")
    model_spec['lambda_w_placeholder'] = tf.placeholder(tf.float32, [], name="lambda_w")
    model_spec['sigma_placeholder'] = tf.placeholder(tf.float32, [], name="sigma_ratio")
    model_spec['gamma_placeholder'] = tf.placeholder(tf.float32, [], name="gamma")

    if mode == 'train':
        model_spec['train_op'] = train_op
        model_spec['train_op_trace_ratio'] = train_op_trace_ratio
    elif mode == 'cluster':
        model_spec['sample'] = new_sample

    return model_spec
