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


def normalized_mutual_information(labels, params, cluster_idx):
    # Calculate the Normalized Mutual Information

    # Get the size of data for calculation
    size_data = len(cluster_idx)

    # Count total number of labels per centroid, using the label of each training
    # sample to their closest centroid (given by 'cluster_idx')
    counts = np.zeros(shape=(params['k'], params['num_classes']))
    for i in range(size_data):
        counts[cluster_idx[i], labels[i]] += 1

    # Calculate Entropy of Class Labels
    counts_labels = tf.reduce_sum(counts, 0)  # column sum
    p_class = counts_labels/size_data
    entropy_class_labels = -p_class*tf.log(p_class)
    entropy_class_labels = tf.reduce_sum(entropy_class_labels)

    # Calculate Entropy of Cluster Indices
    counts_clusters = tf.reduce_sum(counts, 1)  # row sum
    p_cluster = counts_clusters/size_data
    entropy_cluster_idx = -p_cluster*tf.log(p_cluster)
    entropy_cluster_idx = tf.reduce_sum(entropy_cluster_idx)

    # Calculate Conditional Entropy of class labels for clusters
    p_conditional = counts / tf.reshape(counts_clusters, (-1, 1))
    log_p_conditional = tf.log(p_conditional)
    # Setting -inf Values to '0' for correct Calculation of NMI
    log_p_conditional = tf.where(tf.is_inf(log_p_conditional), tf.zeros_like(log_p_conditional), log_p_conditional)
    sum_conditional_entropy = tf.reduce_sum(p_conditional*log_p_conditional, 1)
    conditional_entropy = -p_cluster*sum_conditional_entropy


    # Mutual Information mi
    mi = entropy_class_labels - tf.reduce_sum(conditional_entropy)

    # Normalized Mutual Information nmi
    nmi = (2*mi) / (entropy_class_labels + entropy_cluster_idx)

    return nmi
