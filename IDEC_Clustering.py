import os
import logging

import tensorflow as tf
import numpy as np

from tensorflow.contrib.factorization import KMeansClustering

from utils import save_dict_to_json
from utils import Params
from utils import visualize_embeddings

from metrics import cluster_accuracy
from metrics import normalized_mutual_information
from metrics import adjuster_rand_index


def train_and_evaluate_idec(train_model_spec, model_dir, params, config, restore_from=None, vars_to_restore=None):
    """Train the model and evaluate every epoch.
       Args:
           train_model_spec: (dict) contains the graph operations or nodes needed for training
           eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
           model_dir: (string) directory containing config, weights and log
           params: (Params) contains hyperparameters of the model.
                   Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
           config: (tf.ConfigProto()) Set options for the session (like gpu_options)
           restore_from: (string) directory or file containing weights to restore the graph
       """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver(vars_to_restore)  # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0

    with tf.Session(config=config) as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'], feed_dict={train_model_spec['sigma_placeholder']: params.sigma})

        # Reload weights from directory if specified
        if restore_from is not None:
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
            last_saver.restore(sess, restore_from)

        # For tensorboard (takes care of writing summaries to files)
        writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

        # Load Best eval_loss so far (if existent)
        best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
        # if os.path.isfile(best_json_path):
        #     best_eval_metrics = Params(best_json_path)
        #     best_eval_loss = best_eval_metrics.VAE_loss
        # else:
        best_eval_nmi = 0.0

        # Compute number of batches in one epoch (one full pass over the training set)
        num_steps = (params.eval_size + params.eval_batch_size - 1) // params.eval_batch_size

        # Initialization of centers by running kmeans
        cluster_centers = train_model_spec['cluster_centers']
        kmeans_model = KMeansClustering(num_clusters=params.k, use_mini_batch=True)
        for i in range(20):
            sess.run(train_model_spec['iterator_init_op'])
            kmeans_model.train(lambda: sess.run(train_model_spec['samples'], feed_dict={train_model_spec['sigma_placeholder']:
                                                                                params.sigma}), steps=num_steps)
        sess.run(cluster_centers.assign(kmeans_model.cluster_centers()))

        # Get relevant graph operations or nodes needed for training
        loss = train_model_spec['loss']
        train_op = train_model_spec['train_op']
        update_metrics = train_model_spec['update_metrics']
        metrics = train_model_spec['metrics']
        summary_op = train_model_spec['summary_op']
        global_step = tf.train.get_global_step()

        for epoch in range(begin_at_epoch, (begin_at_epoch + params.num_epochs)):
            # Run one epoch
            if epoch % params.eval_visu_step == 0 or epoch == begin_at_epoch + params.num_epochs - 1:

                sess.run(train_model_spec['metrics_init_op'])
                sess.run(train_model_spec['iterator_init_op'])

                # Update the target distribution for all samples
                # And assign the prediction ((cluster_idx) to each sample
                target_distribution = {}
                y_pred = np.array([])
                labels = np.array([])
                for i in range(num_steps):
                    target_distribution[i], y_pred_batch, labels_batch, img, global_step_val = sess.run([train_model_spec['train_op_distribution'], train_model_spec['cluster_idx'], train_model_spec['labels'], train_model_spec["sample"], global_step],
                                              feed_dict={train_model_spec['sigma_placeholder']: params.sigma,
                                                         train_model_spec['learning_rate_placeholder']: params.initial_training_rate})
                    y_pred = np.append(y_pred, y_pred_batch)
                    labels = np.append(labels, labels_batch)
                    # Input set for TensorBoard visualization
                    if params.visualize == 1:
                        if i == 0:
                            embedded_data = img
                            embedded_labels = labels_batch
                        elif i < 10: # just the embeddings of first 10 batches are saved due to memory restrictions
                            embedded_data = np.concatenate((embedded_data, img), axis=0)
                            embedded_labels = np.concatenate((embedded_labels, labels_batch), axis=0)
                    else:
                        embedded_data = []
                        embedded_labels = []
                
                # Write embeddings to external files for visualizing if desired
                if params.visualize == 1:
                    log_dir = writer.get_logdir()
                    metadata = os.path.join(log_dir, ('metadata' + str(epoch + 1) + '.tsv'))
                    # def save_metadata(file):
                    with open(metadata, 'w') as metadata_file:
                        for c in embedded_labels:
                            metadata_file.write('{}\n'.format(c))
                    visualize_embeddings(sess, log_dir, embedded_data, (epoch + 1), writer, params)
                        
                y_pred = y_pred.astype(int)
                labels = labels.astype(int)
                counts = np.zeros(shape=(params.k, params.num_classes))
                for j in range(len(y_pred)):
                    counts[y_pred[j], labels[j]] += 1
                counts = tf.convert_to_tensor(counts)

                # Assign the most frequent label to the centroid
                labels_map = tf.argmax(counts, axis=1)  # find Label with max. occurrence along each row
                # Evaluation ops
                # Lookup: centroid_id -> label
                y_pred = tf.nn.embedding_lookup(labels_map, y_pred)

                accuracy = sess.run(cluster_accuracy(labels, y_pred))
                nmi = sess.run(normalized_mutual_information(counts))
                ari = sess.run(adjuster_rand_index(counts))

                # Get the values of the metrics
                metrics_values = {k: v[0] for k, v in train_model_spec['metrics'].items()}
                metrics_eval = sess.run(metrics_values)

                metrics_eval['Accuracy'] = accuracy
                metrics_eval['Normalized Mutual Information'] = nmi
                metrics_eval['Adjusted Rand Index'] = ari

                for tag, val in metrics_eval.items():
                    summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
                    writer.add_summary(summ, global_step_val)

                print("Cluster_acc at Epoch ", epoch, ": %.2f" % metrics_eval['Accuracy'])


            # Load the training dataset into the pipeline and initialize the metrics local variables after every epoch
            sess.run(train_model_spec['iterator_init_op'])
            sess.run(train_model_spec['metrics_init_op'])
            for i in range(num_steps):
                # Evaluate summaries for tensorboard only once in a while
                if i % params.save_summary_steps == 0:
                    # Perform a mini-batch update
                    _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                                      summary_op, global_step],
                                                                     feed_dict={train_model_spec['sigma_placeholder']: params.sigma,
                                                                                train_model_spec['learning_rate_placeholder']: params.initial_training_rate,
                                                                                train_model_spec['target_prob']: target_distribution[i],
                                                                                train_model_spec['gamma_placeholder']: params.gamma,
                                                                                train_model_spec['lambda_r_placeholder']: params.lambda_r,
                                                                                train_model_spec['lambda_c_placeholder']: params.lambda_c,
                                                                                train_model_spec['lambda_d_placeholder']: params.lambda_d,
                                                                                train_model_spec['lambda_b_placeholder']: params.lambda_b,
                                                                                train_model_spec['lambda_w_placeholder']: params.lambda_w})
                    # Write summaries for tensorboard
                    writer.add_summary(summ, global_step_val)

                    # Output Training Loss after each summary step
                    print("Training_loss after Step ", global_step_val, ":", loss_val)
                    # print("Step", epoch + 1, "finished -> you are getting closer: %.2f" % ((epoch + 1)/(params.num_epochs*num_steps)), "% done")
                else:
                    _, _ = sess.run([train_op, update_metrics], feed_dict={train_model_spec['sigma_placeholder']: params.sigma,
                                                                           train_model_spec['learning_rate_placeholder']: params.initial_training_rate,
                                                                           train_model_spec['target_prob']: target_distribution[i],
                                                                           train_model_spec['gamma_placeholder']: params.gamma,
                                                                           train_model_spec['lambda_r_placeholder']: params.lambda_r,
                                                                           train_model_spec['lambda_c_placeholder']: params.lambda_c,
                                                                           train_model_spec['lambda_d_placeholder']: params.lambda_d,
                                                                           train_model_spec['lambda_b_placeholder']: params.lambda_b,
                                                                           train_model_spec['lambda_w_placeholder']: params.lambda_w})

            metrics_values = {k: v[0] for k, v in metrics.items()}
            metrics_val = sess.run(metrics_values)
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
            logging.info("- Train metrics: " + metrics_string)
            # Save weights
            #last_save_path = os.path.join(model_dir, 'last_weights/', 'after-epoch')
            #last_saver.save(sess, last_save_path, global_step=epoch + 1)

            if metrics_eval['Accuracy'] >= best_eval_nmi:
                # Save best eval metrics in a json file in the model directory
                save_dict_to_json(metrics_eval, best_json_path)

            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")

            save_dict_to_json(metrics_eval, last_json_path)
