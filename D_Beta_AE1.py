import tensorflow as tf
from tensorflow.contrib.factorization import KMeans


# Define a Leacky ReLu Function
def lrelu(x, alpha=0.2):
    return tf.maximum(x, tf.multiply(x, alpha))


# Defining the Encoder
def encoder(encoder_input, is_training, params):
    x = tf.reshape(encoder_input, shape=[-1, params.resize_height, params.resize_width, params.channels])
    print('-------Encoder-------')
    layer_features = []
    for k in range(4):
        print(x.get_shape())
        x = tf.layers.conv2d(x, filters=16*(2**k), kernel_size=3, strides=1, padding='same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        if k < 3:
            x = tf.layers.max_pooling2d(x, 2, 2)
            layer_features.append(x)

    # Last layer average pooling
    x = tf.layers.average_pooling2d(x, 4, 4)
    print(x.get_shape())
    x = tf.contrib.layers.flatten(x)
    print(x.get_shape())

    z = tf.layers.dense(x, units=params.n_latent, kernel_initializer=tf.contrib.layers.xavier_initializer())
    #z = tf.divide(z,tf.norm(z)+1e-10) # Normalizing l2 norm
    print(z.get_shape())
    print('-------Encoder-------')

    return z


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

    layer_features = []
    for k in range(4):
        x = tf.layers.conv2d_transpose(x, filters=max(16, 16*(2**(3-k-1))), kernel_size=4, strides=2, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print(x.get_shape())
        if k < 3:
            layer_features.append(x)

    reconstructed_mean = tf.layers.conv2d(x, filters=params.channels, kernel_size=3, padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer()) #(activation=tf.nn.sigmoid)tf.reshape(x, shape=[-1, params.resize_height, params.resize_width, params.channels])
    print(reconstructed_mean.get_shape())
    print('-------Decoder-------')
    return reconstructed_mean


def build_model(inputs, is_training, params):

    original_img = inputs["img"]
    # Bringing together Encoder and Decoder
    sampled = encoder(original_img, is_training, params)
    reconstructed_mean = decoder(sampled, is_training, params)

    loss_square = tf.losses.mean_squared_error(labels=original_img, predictions=tf.sigmoid(reconstructed_mean))
    with tf.variable_scope('kmeans'):
        kmeans = KMeans(inputs=sampled, num_clusters=params.k, distance_metric='cosine', use_mini_batch=False,
                          mini_batch_steps_per_iteration=1, initial_clusters='kmeans_plus_plus')
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
        init_op, train_op) = kmeans.training_graph()

    collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='kmeans')
    reset_op = tf.initialize_variables(collection)

    return loss_square, sampled, reconstructed_mean, scores[0], init_op, train_op, reset_op


def b_ae_model_fn(mode, inputs, params, reuse=False):
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
    with tf.variable_scope('b_ae_model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        loss_likelihood, sampled, reconstructed_mean, cluster_center_sim, kmeans_init_op, kmeans_train_op, cluster_reset_op = \
            build_model(inputs, is_training, params)

    # Create Similarity matrix of original images
    # Used for discrimintaive Loss
    z = inputs["img"]/tf.norm(inputs["img"], ord=1)
    z_flat = tf.contrib.layers.flatten(z)
    sim_original_img = tf.matmul(z_flat, z_flat, transpose_b=True)
    sum_anchor = (tf.shape(sampled)[-1])//20  #*0.05 (5%)
    _, anchor_idx = tf.nn.top_k(sim_original_img, k=sum_anchor)

    #tmp=anchor_idx.eval()
    z = sampled/tf.norm(sampled, ord=1)
    C_ij = tf.matmul(z, z, transpose_b=True)
    anchor_val = tf.gather(C_ij, anchor_idx)

    batch_size = (tf.shape(sampled)[-1])
    alpha = tf.constant(0.5)

    L_d = tf.cast((1/(batch_size**2 - sum_anchor)), tf.float32)*tf.cast((tf.reduce_sum(tf.abs(C_ij), [0, 1])-tf.reduce_sum(tf.abs(anchor_val))), tf.float32)
    x = (1.0 - alpha) / tf.cast(sum_anchor, tf.float32)
    y = tf.cast(tf.reduce_sum(tf.abs(anchor_val)), tf.float32)
    L_d = L_d - x*y

    # Sum over all similarities from sample to closest cluster center
    # set the input of this operation (the clusters) as not trainable for the optimizer by stopping gradient here
    L_c = tf.stop_gradient(tf.reduce_sum(cluster_center_sim))
    L_r = loss_likelihood
    lambda_r = tf.placeholder(tf.float32, shape=[], name='Reconstruction_regularization')
    lambda_d = tf.placeholder(tf.float32, shape=[], name='discriminative_regularization')
    lambda_c = tf.placeholder(tf.float32, shape=[], name='cluster_sim_regularization')
    # Define the Loss
    loss = tf.reduce_mean(L_d+L_r)#+lambda_c*L_c)

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
    with tf.variable_scope("b_ae_metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
            #'cluster_center_loss': tf.metrics.mean(sum_cluster_dist),
            'neg_log_likelihood': tf.metrics.mean(loss_likelihood)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="b_ae_metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('neg_log_likelihood', loss_likelihood)
    #tf.summary.scalar('cluster_center_loss', sum_cluster_dist)
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
    model_spec['cluster_center_init'] = kmeans_init_op
    model_spec['learning_rate_placeholder'] = learning_rate_ph
    model_spec['sigma_placeholder'] = tf.placeholder(tf.float32, [], name="sigma_ratio")

    if mode == 'train':
        #train_op = tf.group(*[train_op, kmeans_train_op])
        model_spec['train_op'] = train_op
        model_spec['cluster_center_update'] = kmeans_train_op
        model_spec['cluster_center_reset'] = cluster_reset_op
    elif mode == 'cluster':
        model_spec['sample'] = sampled

    return model_spec
