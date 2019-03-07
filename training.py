import os
import logging

import tensorflow as tf
import numpy as np

from evaluation import evaluate_sess
from utils import save_dict_to_json
from utils import visualize_embeddings
from utils import visualize_umap
from utils import Params


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
                                                              summary_op, global_step],
                                                             feed_dict={model_spec['sigma_placeholder']: params.sigma,
                                                                        model_spec['learning_rate_placeholder']: params.initial_training_rate})#, model_spec['gamma_placeholder']: params.gamma})
            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)

            # Output Training Loss after each summary step
            print("Training_loss after Step ", global_step_val, ":", loss_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss],
                                      feed_dict={model_spec['sigma_placeholder']: params.sigma,
                                                 model_spec['learning_rate_placeholder']: params.initial_training_rate})#, model_spec['gamma_placeholder']: params.gamma})

    # If we have a model with cluster centers in training, update them on training set
    if 'cluster_center_update' in model_spec:
        sess.run(model_spec['cluster_center_reset'])
        sess.run(model_spec['iterator_init_op'])
        for i in range(num_steps):
            sess.run(model_spec['cluster_center_update'])

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)

    return metrics_val


def train_and_evaluate(train_model_spec, eval_model_spec, model_dir, params, config, restore_from=None):
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
    last_saver = tf.train.Saver()  # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0

    with tf.Session(config=config) as sess:
        # Initialize model variables
        sess.run([train_model_spec['variable_init_op'], eval_model_spec['variable_init_op']])

        # If we are using clusters in the train model spec, initialize the cluster centers
        if 'cluster_center_init' in train_model_spec:
            sess.run(train_model_spec['iterator_init_op'])
            sess.run(train_model_spec['cluster_center_init'])
            sess.run(eval_model_spec['iterator_init_op'])
            sess.run(eval_model_spec['cluster_center_init'])

        # Reload weights from directory if specified
        if restore_from is not None:
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

        # Load Best eval_loss so far (if existent)
        best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
        if os.path.isfile(best_json_path):
            best_eval_metrics = Params(best_json_path)
            best_eval_loss = best_eval_metrics.VAE_loss
        else:
            best_eval_loss = 10000.0
            
        metrics_eval = {}
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            if epoch > 10:
                params.gamma = np.minimum(1.0, np.multiply(0.1, (epoch-10.0)))
            # Run one epoch
            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (params.train_size + params.train_batch_size - 1) // params.train_batch_size
            metrics_train = train_sess(sess, train_model_spec, num_steps, train_writer, params)

            # Save weights
            #last_save_path = os.path.join(model_dir, 'last_weights/', 'after-epoch')
            #last_saver.save(sess, last_save_path, global_step=epoch + 1)

            # Do evaulation session just at defined steps or last epoch and after appropriate pretraining
            if epoch % params.eval_visu_step == 0 or epoch == begin_at_epoch + params.num_epochs - 1:
                # Evaluate for one epoch on validation set
                num_steps = (params.eval_size + params.eval_batch_size - 1) // params.eval_batch_size
                metrics_eval, embedded_data, embedded_labels = evaluate_sess(sess, eval_model_spec, num_steps,
                                                                             eval_writer, params)
                print("Cluster_acc after Epoch ", epoch + 1, ": %.2f" % metrics_eval['Accuracy'])
                log_dir = train_writer.get_logdir()
                metadata = os.path.join(log_dir, ('metadata' + str(epoch + 1) + '.tsv'))
                img_latentspace = os.path.join(log_dir, ('latentspace' + str(epoch + 1) + '.txt'))

                np.savetxt(img_latentspace, embedded_data)

                # def save_metadata(file):
                with open(metadata, 'w') as metadata_file:
                    for c in embedded_labels:
                        metadata_file.write('{}\n'.format(c))

            # If best_eval, best_save_path
            metrics_eval['Model_loss'] = metrics_train['loss']

            if metrics_eval['Model_loss'] <= best_eval_loss:
                # Store new best accuracy
                best_eval_loss = metrics_eval['Model_loss']
                # Save weights
                best_save_path = os.path.join(model_dir, 'best_weights/', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                save_dict_to_json(metrics_eval, best_json_path)

            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")

            save_dict_to_json(metrics_eval, last_json_path)

            print("Epoch", epoch + 1, "finished -> you are getting closer: %.1f" % (((epoch + 1)/params.num_epochs)*100), "% done")

        if params.visualize == 1:
            visualize_embeddings(sess, log_dir, train_writer, params)
            visualize_umap(sess, log_dir, train_writer, params)
