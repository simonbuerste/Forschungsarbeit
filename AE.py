import tensorflow as tf


# Define a Leacky ReLu Function
def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


# Defining the Encoder
def encoder(encoder_input, prob_keep, params):
    activation = lrelu
    X = tf.reshape(encoder_input, shape=[-1, params.resize_height, params.resize_width, params.channels])
    x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
    x = tf.nn.dropout(x, prob_keep)
    x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
    x = tf.nn.dropout(x, prob_keep)
    x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
    x = tf.nn.dropout(x, prob_keep)
    x = tf.contrib.layers.flatten(x)
    z = tf.layers.dense(x, units=params.n_latent)

    return z


# Defining the Decoder
def decoder(sampled_z, prob_keep, params):
    reshaped_dim = [-1, params.n_latent, params.n_latent, params.channels]
    inputs_decoder = int(32 * params.channels)

    x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
    x = tf.layers.dense(x, units=inputs_decoder * 2, activation=lrelu)
    x = tf.reshape(x, reshaped_dim)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    x = tf.nn.dropout(x, prob_keep)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
    x = tf.nn.dropout(x, prob_keep)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(x, units=params.resize_height * params.resize_width * params.channels, activation=tf.nn.sigmoid)
    img = tf.reshape(x, shape=[-1, params.resize_height, params.resize_width, params.channels])
    return img


def build_model(inputs, keep_prob, params):

    img = inputs["img"]
    # Bringing together Encoder and Decoder
    sampled = encoder(img, keep_prob, params)
    dec = decoder(sampled, keep_prob, params)

    # Computing Loss and Enforcing a Gaussian Distribution
    unreshaped = tf.reshape(dec, [-1, params.resize_height * params.resize_width * params.channels])
    y_flat = tf.reshape(img, [-1, params.resize_height * params.resize_width * params.channels])
    img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, y_flat), 1)
    return img_loss, sampled


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
    else:
        p_dropout = 1
# -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('ae_model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        img_loss, sampled = build_model(inputs, p_dropout, params)

    # Define the Loss
    loss = tf.reduce_mean(img_loss)

    # Define training step that minimizes the loss with the Adam optimizer
    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(0.0005)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("ae_metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="ae_metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)

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
