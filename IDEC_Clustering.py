import os
import logging

import tensorflow as tf
import numpy as np

from evaluation import evaluate_sess
from utils import save_dict_to_json
from utils import Params

from metrics import cluster_accuracy
from metrics import normalized_mutual_information
from metrics import adjuster_rand_index


def train_sess(sess, model_spec, num_steps, writer, params):
    """Train the model on `num_steps` batches
    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries
        params: (Params) hyperparameters
    """
    # Get relevant graph operations or nodes needed for training
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics local variables
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    for i in range(num_steps):
        # Evaluate summaries for tensorboard only once in a while
        if i % params.save_summary_steps == 0:
            # Perform a mini-batch update
            _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                              summary_op, global_step])
            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)

            # Output Training Loss after each summary step
            print("Training_loss after Step ", global_step_val, ":", loss_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)

    return metrics_val


def train_and_evaluate_idec(train_model_spec, eval_model_spec, model_dir, params, config, restore_from=None, vars_to_restore=None):
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
        # Initialize iterator because cluster center initialization need samples
        sess.run([train_model_spec['iterator_init_op'], eval_model_spec['iterator_init_op']])
        # Initialize model variables
        sess.run([train_model_spec['variable_init_op'], eval_model_spec['variable_init_op']])

        # Reload weights from directory if specified
        if restore_from is not None:
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                #begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # For tensorboard (takes care of writing summaries to files)
        writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

        # Load Best eval_loss so far (if existent)
        best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
        if os.path.isfile(best_json_path):
            best_eval_metrics = Params(best_json_path)
            best_eval_acc = best_eval_metrics.VAE_loss
        else:
            best_eval_acc = 0.0

        # Compute number of batches in one epoch (one full pass over the training set)
        num_steps = (params.train_size + params.train_batch_size - 1) // params.train_batch_size

        # Get relevant graph operations or nodes needed for training
        loss = train_model_spec['loss']
        train_op = train_model_spec['train_op']
        update_metrics = train_model_spec['update_metrics']
        metrics = train_model_spec['metrics']
        summary_op = train_model_spec['summary_op']
        global_step = tf.train.get_global_step()

        for epoch in range(begin_at_epoch, (begin_at_epoch + params.num_epochs)):
            # Run one epoch
            if epoch % params.eval_visu_step == 0:
                num_steps_eval = (params.eval_size + params.eval_batch_size - 1) // params.eval_batch_size

                accuracy = 0
                nmi = 0
                ari = 0
                sess.run(eval_model_spec['metrics_init_op'])
                sess.run(eval_model_spec['iterator_init_op'])
                for i in range(num_steps_eval):
                    _, y_pred, labels = sess.run([eval_model_spec['train_op_distribution'],
                                                     eval_model_spec['cluster_idx'], eval_model_spec['labels']])

                    counts = np.zeros(shape=(params.k, params.num_classes))
                    for j in range(len(y_pred)):
                        counts[y_pred[j], labels[j]] += 1
                    counts = tf.convert_to_tensor(counts)

                    accuracy += sess.run(cluster_accuracy(labels, y_pred))
                    nmi += sess.run(normalized_mutual_information(counts))
                    ari += sess.run(adjuster_rand_index(counts))

                # Get the values of the metrics
                metrics_values = {k: v[0] for k, v in eval_model_spec['metrics'].items()}
                metrics_eval = sess.run(metrics_values)

                metrics_eval['Accuracy'] = accuracy / num_steps_eval
                metrics_eval['Normalized Mutual Information'] = nmi / num_steps_eval
                metrics_eval['Adjusted Rand Index'] = ari / num_steps_eval
                print("Cluster_acc at Epoch ", epoch, ": %.2f" % metrics_eval['Accuracy'])

            # Load the training dataset into the pipeline and initialize the metrics local variables after every epoch

            sess.run(train_model_spec['iterator_init_op'])
            sess.run(train_model_spec['metrics_init_op'])
            for i in range(num_steps):
                # Evaluate summaries for tensorboard only once in a while
                if i % params.save_summary_steps == 0:
                    # Perform a mini-batch update
                    _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                                      summary_op, global_step])
                    # Write summaries for tensorboard
                    writer.add_summary(summ, global_step_val)

                    # Output Training Loss after each summary step
                    print("Training_loss after Step ", global_step_val, ":", loss_val)
                    # print("Step", epoch + 1, "finished -> you are getting closer: %.2f" % ((epoch + 1)/(params.num_epochs*num_steps)), "% done")
                else:
                    _, _, loss_val = sess.run([train_op, update_metrics, loss])

            metrics_values = {k: v[0] for k, v in metrics.items()}
            metrics_val = sess.run(metrics_values)
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
            logging.info("- Train metrics: " + metrics_string)
            # Save weights
            #last_save_path = os.path.join(model_dir, 'last_weights/', 'after-epoch')
            #last_saver.save(sess, last_save_path, global_step=epoch + 1)

            if metrics_eval['Accuracy'] >= best_eval_acc:
                # Save best eval metrics in a json file in the model directory
                save_dict_to_json(metrics_eval, best_json_path)

            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")

            save_dict_to_json(metrics_eval, last_json_path)
