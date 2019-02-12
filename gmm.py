import tensorflow as tf
from tensorflow.contrib.factorization import gmm


def build_gmm_model(inputs, params):
    imgs = inputs["img"]

    training_graph = gmm(inp=imgs, initial_clusters='random', num_clusters=params.k, random_seed=42,
                         covariance_type="diag", params="wmc")

    return training_graph


def gmm_model_fn(inputs, params, reuse=False):

    with tf.variable_scope('gmm', reuse=reuse):
        # Build up the kMeans Model
        # K-Means Parameters
        training_graph = build_gmm_model(inputs, params)

    # loss - Returns the log-likelihood operation
    # scores - Log probabilities of each data point
    # cluster_idx - Returns a list of Tensors with the matrix of assignments per shard
    loss, scores, cluster_idx, train_op, init_op, cluster_centers_initialized = training_graph

    global_step = tf.train.get_or_create_global_step()
    cluster_idx = cluster_idx[0][0]  # fix for cluster_idx being a list of tuple

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)

    with tf.variable_scope("kmeans_metrics"):
        metrics = {
            'score': tf.metrics.mean(scores)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="kmeans_metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('score', scores)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['score'] = scores
    model_spec['cluster_idx'] = cluster_idx
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['train_op'] = train_op
    model_spec['init_op'] = init_op
    model_spec['cluster_center_initialized'] = cluster_centers_initialized

    return model_spec
