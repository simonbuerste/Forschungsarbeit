import numpy as np
import tensorflow as tf


def cluster_accuracy(labels, cluster_label):
    # Calculate Accuracy of Clustering

    # Compute accuracy
    correct_prediction = tf.equal(cluster_label, tf.cast(labels, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def normalized_mutual_information(counts):
    # Calculate the Normalized Mutual Information

    # Get the size of data for calculation
    size_data = tf.reduce_sum(counts)

    # Calculate Entropy of Class Labels
    counts_labels = tf.reduce_sum(counts, 0)  # column sum
    p_class = counts_labels/size_data
    entropy_classes = -p_class*tf.log(p_class)
    # Setting -inf Values to '0' for correct Calculation of NMI
    entropy_classes = tf.where(tf.logical_or(tf.is_inf(entropy_classes), tf.is_nan(entropy_classes)),
                                        tf.zeros_like(entropy_classes), entropy_classes)
    entropy_classes = tf.reduce_sum(entropy_classes)

    # Calculate Entropy of Cluster Indices
    counts_clusters = tf.reduce_sum(counts, 1)  # row sum
    p_cluster = counts_clusters/size_data
    entropy_clusters = -p_cluster*tf.log(p_cluster)
    # Setting -inf Values to '0' for correct Calculation of NMI
    entropy_clusters = tf.where(tf.logical_or(tf.is_inf(entropy_clusters), tf.is_nan(entropy_clusters)),
                                       tf.zeros_like(entropy_clusters), entropy_clusters)
    entropy_clusters = tf.reduce_sum(entropy_clusters)

    # Calculate Conditional Entropy of class labels for clusters
    p_conditional = counts / tf.reshape(counts_clusters, (-1, 1))
    log_p_conditional = p_conditional*tf.log(p_conditional)
    # Setting -inf Values to '0' for correct Calculation of NMI
    log_p_conditional = tf.where(tf.logical_or(tf.is_inf(log_p_conditional), tf.is_nan(log_p_conditional)),
                                 tf.zeros_like(log_p_conditional), log_p_conditional)
    sum_conditional_entropy = tf.reduce_sum(log_p_conditional, 1)
    conditional_entropy = -p_cluster*sum_conditional_entropy

    # Mutual Information mi
    mi = entropy_classes - tf.reduce_sum(conditional_entropy)

    # Normalized Mutual Information nmi
    nmi = (2*mi) / (entropy_classes + entropy_clusters)

    return nmi


def adjuster_rand_index(counts):
    # Calculate the adjusted Rand Index
    # For better understanding of variable's naming see Dave Tang's Blog over ARI

    # Calculate binomial coefficients of all counts over 2
    # Variable called "n"
    n_enum = tf.exp(tf.lgamma(counts+1))  # (add + 1 due to definition of gamma function)
    n_denom = 2*tf.exp(tf.lgamma(counts-1))  # (add + 1 due to definition of gamma function)
    n_bi_coeff = n_enum/n_denom
    n_sum_bi_coeff = tf.reduce_sum(n_bi_coeff)

    # Calculate binomial coefficients of the sums of each cluster over 2
    # Variabel called "a"
    a = tf.reduce_sum(counts, 1)  # row sum
    a_enum = tf.exp(tf.lgamma(a+1))  # (add + 1 due to definition of gamma function)
    a_denom = 2*tf.exp(tf.lgamma(a-1))  # (add + 1 due to definition of gamma function)
    a_bi_coeff = a*(a-1)/2
    a_sum_bi_coeff = tf.reduce_sum(a_bi_coeff)

    # Calculate binomial coefficients of the sums of each labels over 2
    # Variabel called "b"
    b = tf.reduce_sum(counts, 0)  # column sum
    #b_enum = tf.exp(tf.lgamma(b+1))  # (add + 1 due to definition of gamma function)
    #b_denom = 2*tf.exp(tf.lgamma(b-1))  # (add + 1 due to definition of gamma function)
    b_bi_coeff = b*(b-1)/2
    b_sum_bi_coeff = tf.reduce_sum(b_bi_coeff)

    # Calculate number of samples
    no_samples = tf.reduce_sum(counts)
    no_samples_2_bicoeff = no_samples*(no_samples-1)/2  # binomial coefficient of no_samples over 2

    debug1 = a_bi_coeff.eval()
    debug2 = b_bi_coeff.eval()
    debug4 = no_samples_2_bicoeff.eval()

    # Calculate enumerator and denominator of ARI
    ari_enum = n_sum_bi_coeff - (a_sum_bi_coeff*b_sum_bi_coeff)/no_samples_2_bicoeff
    ari_denom = 0.5*(a_sum_bi_coeff+b_sum_bi_coeff) - (a_sum_bi_coeff*b_sum_bi_coeff)/no_samples_2_bicoeff
    # Calculate ARI
    ari = ari_enum/ari_denom

    return ari
