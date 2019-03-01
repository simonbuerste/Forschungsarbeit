import tensorflow as tf
from tensorflow.contrib.factorization import KMeansClustering


def student_t_distr(imgs, cluster_centers):
    #q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(imgs, axis=1) - cluster_centers), axis=2)))
    #q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, 1))
    df = tf.constant(1.0)
    scale = tf.constant(1.0)
    distr = tf.distributions.StudentT(df=df, loc=cluster_centers, scale=scale)
    q = distr.prob(imgs)
    return q


def target_distr(q):
    weight = q**2 / tf.reduce_sum(q, 0)
    p = tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, 1))
    return p


def init_centroids(imgs, params):
    # 1st Step: Initialize centroids randomly by samples
    n_samples = tf.shape(imgs)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples))
    begin = [0, ]
    size = [params.k, ]
    size[0] = params.k
    centroid_indices = tf.slice(random_indices, begin, size)
    centroids = tf.gather(imgs, centroid_indices)

    # Do assigning and cluster updating for one run of training samples

    # 2nd Step: Assign samples to centroids
    expanded_vectors = tf.expand_dims(imgs, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    nearest_indices = tf.argmin(distances, 0)

    # 3rd Step: Update Cluster centers
    nearest_indices = tf.to_int32(nearest_indices)
    partitions = tf.dynamic_partition(imgs, nearest_indices, params.k)
    centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)

    return centroids


def build_idec_model(inputs, params):
    imgs = inputs["samples"]

    # Initialization of centers by running kmeans
    # kmeans_model = KMeansClustering(num_clusters=params.k, use_mini_batch=False)
    # kmeans_model.train(input_fn(imgs), steps=20)
    # kmeans_model.cluster_centers()
    cluster_centers = tf.get_variable(name='cluster_centers', initializer=init_centroids(imgs, params))

    q = student_t_distr(imgs, cluster_centers)
    p = target_distr(q)

    kl_loss = tf.multiply(p, tf.log(q / p))
    kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))

    return kl_loss, q, p


def idec_model_fn(inputs, latent_model_spec, params, reuse=False):

    with tf.variable_scope('IDEC', reuse=reuse):
        # Build up the kMeans Model
        # K-Means Parameters
        kl_loss, q, train_op_distr = build_idec_model(inputs, params)

        cluster_idx = tf.argmax(q, axis=1)

        loss = latent_model_spec["loss"] + params.gamma*kl_loss

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
    tf.summary.scalar('loss', loss)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['cluster_idx'] = cluster_idx
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['train_op_distribution'] = train_op_distr
    model_spec['train_op'] = train_op

    return model_spec
