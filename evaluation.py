"""Tensorflow utility functions for evaluation"""

import logging
import os

import tensorflow as tf
import numpy as np

from utils import save_dict_to_json
from metrics import cluster_accuracy
from metrics import normalized_mutual_information
from metrics import adjuster_rand_index


def evaluate_sess(sess, model_spec, num_steps, writer=None, params=None, epoch=None):
    """Evaluate the model on `num_steps` batches.
    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
        epoch:  (int)   Epoch which is currently running (Necessary for correct writing of summary in case no optimizer used)
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])

    # Initialize the cluster centers for each epoch (Cluster Variables have to be initialized ("reset") again therefore)
    sess.run(model_spec['reset_op'])
    while not sess.run(model_spec['cluster_center_initialized']):
        sess.run(model_spec['init_op'], feed_dict={model_spec['sigma_placeholder']: params.sigma})

    sess.run(model_spec['metrics_init_op'])
    sess.run(model_spec['iterator_init_op'])  # 2nd initialization necessary in Case of batch_size = size_of_data

    # train the clustering algorithm
    for _ in range(params.cluster_training_epochs):
        for _ in range(num_steps):
            sess.run(model_spec['train_op'], feed_dict={model_spec['sigma_placeholder']: params.sigma})
        sess.run(model_spec['iterator_init_op'])

    # compute metrics over the dataset
    ypred = np.array([])
    labels = np.array([])
    for i in range(num_steps):
        _, idx_batch, labels_batch, img = sess.run(
            [update_metrics, model_spec['cluster_idx'], model_spec["labels"],
             model_spec["sample"]], feed_dict={model_spec['sigma_placeholder']: params.sigma,
                                               model_spec['lambda_r_placeholder']: params.lambda_r,
                                               model_spec['lambda_c_placeholder']: params.lambda_c,
                                               model_spec['lambda_d_placeholder']: params.lambda_d,
                                               model_spec['lambda_b_placeholder']: params.lambda_b,
                                               model_spec['lambda_w_placeholder']: params.lambda_w})

        ypred = np.append(ypred, idx_batch)
        labels = np.append(labels, labels_batch)

        # Input set for TensorBoard visualization
        if params.visualize == 1:
            if i == 0:
                embedded_data = img
                embedded_labels = labels_batch
            else:
                embedded_data = np.concatenate((embedded_data, img), axis=0)
                embedded_labels = np.concatenate((embedded_labels, labels_batch), axis=0)
        else:
            embedded_data = []
            embedded_labels = []

    ypred = ypred.astype(int)
    labels = labels.astype(int)
    # Evaluate

    # Assign a label to each centroid
    # Count total number of labels per centroid, using the label of each training
    # sample to their closest centroid (given by 'cluster_idx')
    counts = np.zeros(shape=(params.k, params.num_classes))
    for j in range(len(ypred)):
        counts[ypred[j], labels[j]] += 1
    counts = tf.convert_to_tensor(counts)

    # Assign the most frequent label to the centroid
    labels_map = tf.argmax(counts, axis=1)  # find Label with max. occurrence along each row

    # Evaluation ops
    # Lookup: centroid_id -> label
    y_pred = tf.nn.embedding_lookup(labels_map, ypred)

    accuracy = sess.run(cluster_accuracy(labels, y_pred))
    nmi = sess.run(normalized_mutual_information(counts))
    ari = sess.run(adjuster_rand_index(counts))

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)

    metrics_val['Accuracy'] = accuracy
    metrics_val['Normalized Mutual Information'] = nmi
    metrics_val['Adjusted Rand Index'] = ari

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)

    # Add summaries manually to writer at global_step_val
    if writer is not None:
        if epoch is not None:
            global_step_val = epoch
        else:
            global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)

    return metrics_val, embedded_data, embedded_labels


def evaluate(model_spec, model_dir, params, restore_from, config, saver):
    """Evaluate the model
    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
        config: (Configuration) configuration for the session (GPU_options)
        saver: (tf.Train.Saver) saver with defined variables to restore
    """

    with tf.Session(config=config) as sess:
        # Initialize the variables
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        cluster_writer = tf.summary.FileWriter(os.path.join(model_dir, 'cluster_summaries'), sess.graph)

        num_steps = (params.eval_size + params.eval_batch_size - 1) // params.eval_batch_size

        best_test_nmi = 0.0
        for epoch in range(params.num_epochs):
            metrics, _, _ = evaluate_sess(sess, model_spec, num_steps, cluster_writer, params, epoch)

            # If best_eval, best_save_path
            test_nmi = metrics['Normalized Mutual Information']
            if test_nmi >= best_test_nmi:
                # Store new best accuracy
                best_test_acc = test_nmi
                metrics_name = '_'.join(restore_from.split('/'))
                save_path = os.path.join(model_dir, "metrics_test.json")
                save_dict_to_json(metrics, save_path)

            print("Test_acc after Epoch ", epoch+1, ":", metrics['Accuracy'])
