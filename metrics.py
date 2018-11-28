import numpy as np
import tensorflow as tf


def cluster_accuracy(labels, params, cluster_idx):
    # Calculate Accuracy of Clustering
    # Assign a label to each centroid
    # Count total number of labels per centroid, using the label of each training
    # sample to their closest centroid (given by 'cluster_idx')
    counts = np.zeros(shape=(params['k'], params['num_classes']))
    for i in range(len(cluster_idx)):
        counts[cluster_idx[i], labels[i]] += 1
    # Assign the most frequent label to the centroid
    labels_map = [np.argmax(c) for c in counts]
    labels_map = tf.convert_to_tensor(labels_map)
    # Evaluation ops
    # Lookup: centroid_id -> label
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
    # Compute accuracy
    correct_prediction = tf.equal(cluster_label, tf.cast(labels, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
