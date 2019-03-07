import tensorflow as tf


def student_t_distr(imgs, cluster_centers):
    df = tf.constant(1.0)
    y = (1 + tf.reduce_sum(tf.squared_difference(tf.expand_dims(imgs, axis=1), cluster_centers), axis=2))**(-0.5*(df+1))
    q = tf.transpose(tf.transpose(y) / tf.reduce_sum(y, axis=1))
    return q


def target_distr(q):
    weight = q**2 / tf.reduce_sum(q, 0)
    p = tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, 1))
    return p


def build_idec_model(inputs, params):
    imgs = inputs["samples"]

    cluster_centers = tf.get_variable(name='cluster_centers', shape=(params.k, params.n_latent), initializer=tf.zeros_initializer)  # init_centroids(imgs, params))

    q = student_t_distr(imgs, cluster_centers)
    p = tf.placeholder(tf.float32, shape=(None, params.k), name="target_distr")# tf.get_variable(name='target_prob', initializer=tf.zeros_initializer, shape=(params.train_batch_size, params.k), trainable=False)
    #p = tf.reshape(p, [-1, params.k])
    update_target_distr = target_distr(q)

    kl_loss = tf.multiply(p, tf.log(q / p))
    kl_loss = -tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))

    return kl_loss, q, p, update_target_distr, cluster_centers


def idec_model_fn(inputs, latent_model_spec, params, reuse=False):

    with tf.variable_scope('IDEC', reuse=reuse):
        # Build up the kMeans Model
        # K-Means Parameters
        kl_loss, q, p, train_op_distr, cluster_centers = build_idec_model(inputs, params)

        cluster_idx = tf.argmax(q, axis=1)

        reconstruction_loss = latent_model_spec['loss']# tf.reduce_sum(tf.squared_difference(latent_model_spec['img'], latent_model_spec['reconstructions']), axis=[0, 1, 2, 3])
        loss = tf.reduce_mean(reconstruction_loss + params.gamma*kl_loss)

        optimizer = tf.train.AdamOptimizer(params.initial_training_rate)
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)

    with tf.variable_scope("IDEC_metrics"):
        metrics = {
            'clustering_loss': tf.metrics.mean(kl_loss),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="IDEC_metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('clustering_loss', kl_loss)
    tf.summary.scalar('loss_IDEC', loss)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['cluster_idx'] = cluster_idx
    model_spec['cluster_centers'] = cluster_centers
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['train_op_distribution'] = train_op_distr
    model_spec['train_op'] = train_op
    model_spec['loss'] = loss
    model_spec['prob'] = q
    model_spec['target_prob'] = p

    return model_spec
