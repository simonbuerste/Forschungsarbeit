import tensorflow as tf
from tensorflow.contrib.factorization import KMeans


def sample_gumbel(shape, eps=1e-20):
    u = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(u + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    return y


def build_argmax_model(inputs, model_spec, params):
    samples = inputs["samples"]

    cluster_centers = model_spec['cluster_centers']
    cluster_centers_normed = cluster_centers/tf.norm(cluster_centers, ord=1)
    z = samples / tf.norm(samples, ord=1)
    cluster_center_sim = tf.matmul(z, cluster_centers_normed, transpose_b=True)
    assignment = tf.argmax(cluster_center_sim, axis=1)

    # tau = tf.constant(1.0)
    # prob = gumbel_softmax(imgs, tau)

    #prob = imgs
    #assignment = tf.argmax(prob, axis=1)

    return assignment


def argmax_model_fn(inputs, model_spec, params, reuse=False):

    with tf.variable_scope('argmax', reuse=reuse):
        # Build up the kMeans Model
        # K-Means Parameters
        argmax_op = build_argmax_model(inputs, model_spec, params)

    global_step = tf.train.get_or_create_global_step()

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['train_op'] = tf.no_op()
    model_spec['reset_op'] = tf.no_op()
    model_spec['cluster_center_initialized'] = tf.constant(1)
    model_spec['cluster_idx'] = argmax_op

    return model_spec
