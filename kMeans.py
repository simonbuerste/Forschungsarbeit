import tensorflow as tf
from tensorflow.contrib.factorization import KMeans


def build_kmeans_model(inputs, params):
    imgs = inputs["samples"]

    kmeans_model = KMeans(inputs=imgs, num_clusters=params.k, distance_metric='squared_euclidean', use_mini_batch=False,
                          mini_batch_steps_per_iteration=1, initial_clusters='kmeans_plus_plus')

    # Build KMeans graph
    training_graph = kmeans_model.training_graph()

    return training_graph


def kmeans_model_fn(inputs, params, reuse=False):

    with tf.variable_scope('kmeans', reuse=reuse):
        # Build up the kMeans Model
        # K-Means Parameters
        training_graph = build_kmeans_model(inputs, params)

    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph

    global_step = tf.train.get_or_create_global_step()
    cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['score'] = scores
    model_spec['cluster_idx'] = cluster_idx
    model_spec['train_op'] = train_op
    model_spec['init_op'] = init_op
    model_spec['cluster_center_initialized'] = cluster_centers_initialized

    return model_spec
