import tensorflow as tf
from tensorflow.contrib.factorization import KMeans


# Define a Leacky ReLu Function
def lrelu(x, alpha=0.2):
    return tf.maximum(x, tf.multiply(x, alpha))


# Defining the Encoder
def encoder(encoder_input, is_training, params):
    x = tf.reshape(encoder_input, shape=[-1, params.resize_height, params.resize_width, params.channels])
    print('-------Encoder-------')
    print(x.get_shape())
    x = tf.layers.conv2d(x, filters=16, kernel_size=5, strides=2, padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    print(x.get_shape())
    x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    print(x.get_shape())
    x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    print(x.get_shape())
    x = tf.layers.conv2d(x, filters=128, kernel_size=2, strides=2, padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    print(x.get_shape())
    #x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.layers.conv2d(x, filters=128, kernel_size=2, strides=2, padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.leaky_relu(x, alpha=0.2)

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

    reshaped_dim = [-1, params.resize_height//32, params.resize_width//32, 128]
    inputs_decoder = int((params.resize_height//32)*(params.resize_width//32)*128)
    x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
    print(x.get_shape())
    x = tf.reshape(x, reshaped_dim)
    print(x.get_shape())

    # x = selfattentionlayer(x, 'decoder_%d' % (k+1), sigma)
    # x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=2, strides=2, padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    print(x.get_shape())
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=2, strides=2, padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    print(x.get_shape())
    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=3, strides=2, padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    print(x.get_shape())
    x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=5, strides=2, padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    print(x.get_shape())
    reconstructed_mean = tf.layers.conv2d_transpose(x, filters=params.channels, kernel_size=5, strides=2, padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())

    #reconstructed_mean = tf.layers.conv2d_transpose(x, filters=params.channels, kernel_size=4, strides=2, padding='same',
    #                                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    print(reconstructed_mean.get_shape())
    print('-------Decoder-------')
    return reconstructed_mean


def build_model(inputs, is_training, params):

    original_img = inputs["img"]
    # Bringing together Encoder and Decoder
    sampled = encoder(original_img, is_training, params)
    reconstructed_mean = decoder(sampled, is_training, params)

    #dyn_rage = tf.reduce_max(original_img) - tf.reduce_min(original_img)
    #ssim_loss = tf.reduce_mean(1 - (tf.image.ssim(original_img, tf.sigmoid(reconstructed_mean), max_val=dyn_rage)+1.0)/2.0)  # bring it to 0-1 format
    loss_square = tf.losses.mean_squared_error(labels=original_img, predictions=tf.sigmoid(reconstructed_mean))
    #loss_square = tf.norm(original_img-tf.sigmoid(reconstructed_mean))

    return loss_square, sampled, reconstructed_mean


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
        loss_likelihood, sampled, reconstructed_mean = build_model(inputs, is_training, params)
        L_kl_matrix = tf.get_variable(name='L_kl_matrix', shape=(params.k, params.k), initializer=tf.zeros_initializer)
        L_w_matrix = tf.get_variable(name='L_w_matrix', shape=params.k, initializer=tf.zeros_initializer)

    with tf.variable_scope('cluster_center', reuse=reuse):
        cluster_centers = tf.get_variable(name='cluster_centers', shape=(params.k, params.n_latent),
                                          initializer=tf.glorot_uniform_initializer)

    # Create Similarity matrix of original images
    # Used for discrimintaive Loss

    z = tf.reshape(inputs["img"], shape=[-1, params.resize_height*params.resize_width*params.channels])
    z_flat = z/tf.norm(z, ord=1)
    sim_original_img = tf.matmul(z_flat, z_flat, transpose_b=True) # tf.tensordot(z, tf.transpose(z), axes=3)
    #dyn_rage = tf.reduce_max(z) - tf.reduce_min(z)
    #sim_original_img = tf.image.ssim(z, z, max_val=dyn_rage)
    sum_anchor = (tf.shape(sampled)[-1])//params.k
    _, anchor_idx = tf.nn.top_k(sim_original_img, k=sum_anchor)

    z = sampled / tf.norm(sampled, ord=1) # tf.reshape(tf.reduce_sum(sampled, axis=1), (-1, 1))
    C_ij = tf.matmul(z, z, transpose_b=True)
    anchor_val = tf.gather(C_ij, anchor_idx)

    batch_size = tf.shape(sampled)[-1]
    sum_anchor_values = tf.size(anchor_idx)
    alpha = tf.constant(0.5)

    L_d = tf.cast((1/(batch_size**2 - sum_anchor_values)), tf.float32)*tf.cast((tf.reduce_sum(tf.abs(C_ij))-tf.reduce_sum(tf.abs(anchor_val))), tf.float32)
    x = (1.0 - alpha) / tf.cast(sum_anchor_values, tf.float32)
    y = tf.cast(tf.reduce_sum(anchor_val), tf.float32)
    L_d = L_d - x*y

    # Sum over all similarities from sample to closest cluster center
    # set the input of this operation (the clusters) as not trainable for the optimizer by stopping gradient here
    cluster_centers_normed = cluster_centers/tf.norm(cluster_centers, ord=1)
    cluster_center_sim = tf.matmul(z, cluster_centers_normed, transpose_b=True)
    assignment = tf.stop_gradient(tf.one_hot(tf.argmax(cluster_center_sim, axis=1), params.k))
    L_c = tf.reduce_mean(assignment*cluster_center_sim)
    L_r = loss_likelihood

    for i in range(params.k):
        idx_i = tf.cast(tf.where(tf.argmax(cluster_center_sim, axis=1) == i), tf.int32)
        C_i = tf.gather_nd(C_ij, idx_i)
        for j in range(params.k):
            idx_j = tf.cast(tf.where(tf.argmax(cluster_center_sim, axis=1) == j), tf.int32)
            cluster_ij = tf.gather(C_i, idx_j, axis=1)

            kardinality_cluster_i = tf.size(idx_i, out_type=tf.float32)
            kardinality_cluster_j = tf.size(idx_j, out_type=tf.float32)
            denum = kardinality_cluster_i*kardinality_cluster_j
            if i == j:
                L_w_matrix[i].assign((1/denum)*tf.reduce_sum(cluster_ij))
            else:
                L_kl_matrix[i, j].assign((1/denum)*tf.reduce_sum(tf.abs(cluster_ij)))

    L_w = tf.reduce_mean(L_w_matrix)
    L_b = tf.reduce_max(L_kl_matrix)

    lambda_r = tf.placeholder(tf.float32, shape=[], name='Reconstruction_regularization')
    lambda_d = tf.placeholder(tf.float32, shape=[], name='discriminative_regularization')
    lambda_c = tf.placeholder(tf.float32, shape=[], name='center_sim_regularization')
    lambda_b = tf.placeholder(tf.float32, shape=[], name='inter_cluster_sim_regularization')
    lambda_w = tf.placeholder(tf.float32, shape=[], name='intra_cluster_sim_regularization')
    # Define the Loss
    loss = tf.abs(tf.reduce_mean(lambda_c*L_c+lambda_d*L_d+lambda_r*L_r+lambda_b*L_b+lambda_w*L_w))

    # Define training step that minimizes the loss with the Adam optimizer
    learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate")
    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate_ph)
        global_step = tf.train.get_or_create_global_step()
        vars_ae_train_op = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "b_ae_model")
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_ae_train_op)
            train_op_c = optimizer.minimize(-L_c, global_step=global_step, var_list=cluster_centers)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("b_ae_metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
            'discriminative_loss': tf.metrics.mean(L_d),
            'cluster_sample_similarity': tf.metrics.mean(L_c),
            'between_cluster_similarity': tf.metrics.mean(L_b),
            'within_cluster_similarity': tf.metrics.mean(L_w),
            'neg_log_likelihood': tf.metrics.mean(loss_likelihood)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="b_ae_metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('discriminative_loss', L_d)
    tf.summary.scalar('cluster_sample_similarity', L_c)
    tf.summary.scalar('between_cluster_similarity', L_b)
    tf.summary.scalar('within_cluster_similarity', L_w)
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
    model_spec['cluster_centers'] = cluster_centers
    model_spec['learning_rate_placeholder'] = learning_rate_ph
    model_spec['lambda_r_placeholder'] = lambda_r
    model_spec['lambda_c_placeholder'] = lambda_c
    model_spec['lambda_d_placeholder'] = lambda_d
    model_spec['lambda_b_placeholder'] = lambda_b
    model_spec['lambda_w_placeholder'] = lambda_w
    model_spec['sigma_placeholder'] = tf.placeholder(tf.float32, [], name="sigma_ratio")
    model_spec['gamma_placeholder'] = tf.placeholder(tf.float32, [], name="gamma")

    if mode == 'train':
        #train_op = tf.group(*[train_op, kmeans_train_op])
        model_spec['train_op'] = train_op
        model_spec['cluster_center_update'] = train_op_c
        model_spec['samples'] = sampled
    elif mode == 'cluster':
        model_spec['sample'] = sampled

    return model_spec
