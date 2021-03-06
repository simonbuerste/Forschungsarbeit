import os
import logging

import tensorflow as tf
import numpy as np

from tensorflow.contrib.factorization import KMeansClustering

from evaluation import evaluate_sess
from utils import save_dict_to_json
from utils import visualize_embeddings
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
    if "train_op_c" in model_spec:
        train_op = tf.group(*[model_spec['train_op'], model_spec['train_op_c']])
    # elif "train_op_trace_ratio" in model_spec:
    #     train_op = tf.group(*[model_spec['train_op'], model_spec['train_op_trace_ratio']])
    else:
        train_op = model_spec['train_op']

    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics local variables
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # If we have a model with cluster centers in training, update them on training set
    if 'cluster_center_update' in model_spec:
        train_op_additional = model_spec['cluster_center_update']
        # sess.run(model_spec['cluster_center_reset'])
        # sess.run(model_spec['cluster_center_init'])
    elif 'train_op_trace_ratio' in model_spec:
        train_op_additional = model_spec['train_op_trace_ratio']
    else:
        train_op_additional = tf.no_op()

    for i in range(num_steps):
        sess.run(train_op_additional,
                 feed_dict={model_spec['sigma_placeholder']: params.sigma,
                            model_spec['learning_rate_placeholder']: params.initial_training_rate,
                            model_spec['lambda_r_placeholder']: params.lambda_r,
                            model_spec['lambda_c_placeholder']: params.lambda_c,
                            model_spec['lambda_d_placeholder']: params.lambda_d,
                            model_spec['lambda_b_placeholder']: params.lambda_b,
                            model_spec['lambda_w_placeholder']: params.lambda_w})
    sess.run(model_spec['iterator_init_op'])

    for i in range(num_steps):
        # Evaluate summaries for tensorboard only once in a while
        if i % params.save_summary_steps == 0:
            # Perform a mini-batch update
            _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                              summary_op, global_step],
                                                             feed_dict={model_spec['sigma_placeholder']: params.sigma,
                                                                        model_spec['learning_rate_placeholder']: params.initial_training_rate,
                                                                        model_spec['lambda_r_placeholder']: params.lambda_r,
                                                                        model_spec['lambda_c_placeholder']: params.lambda_c,
                                                                        model_spec['lambda_d_placeholder']: params.lambda_d,
                                                                        model_spec['lambda_b_placeholder']: params.lambda_b,
                                                                        model_spec['lambda_w_placeholder']: params.lambda_w})
            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)

            # Output Training Loss after each summary step
            print("Training_loss after Step ", global_step_val, ":", loss_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss],
                                      feed_dict={model_spec['sigma_placeholder']: params.sigma,
                                                 model_spec['learning_rate_placeholder']: params.initial_training_rate,
                                                 model_spec['lambda_r_placeholder']: params.lambda_r,
                                                 model_spec['lambda_c_placeholder']: params.lambda_c,
                                                 model_spec['lambda_d_placeholder']: params.lambda_d,
                                                 model_spec['lambda_b_placeholder']: params.lambda_b,
                                                 model_spec['lambda_w_placeholder']: params.lambda_w})

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
        sess.run([train_model_spec['iterator_init_op'], eval_model_spec['iterator_init_op']])
        sess.run([train_model_spec['variable_init_op'], eval_model_spec['variable_init_op']])

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
        # if os.path.isfile(best_json_path):
        #     best_eval_metrics = Params(best_json_path)
        #     best_eval_loss = best_eval_metrics.VAE_loss
        # else:
        best_eval_nmi = 0.0
            
        metrics_eval = {}
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            # Save the "starting" point of learning rate
            if epoch == 0:
                learning_rate_start = params.initial_training_rate

            if params.learning_rate_schedule == "step_decay":
                if epoch == 20:
                    params.initial_training_rate = params.initial_training_rate / params.decay_factor
                elif epoch == 40:
                    params.initial_training_rate = params.initial_training_rate / params.decay_factor
            elif params.learning_rate_schedule == "exponential_decay":
                if epoch == 0:
                    params.decay_factor = -np.log(1 / (params.decay_factor ** 2)) / (params.num_epochs - params.t0)
                if epoch >= params.t0:
                    params.initial_training_rate = learning_rate_start * np.exp(
                        -params.decay_factor * (epoch - params.t0))
            elif params.learning_rate_schedule == "triangular":
                if epoch == 0:
                    base_lr = learning_rate_start / (params.decay_factor ** 2)
                    max_lr = learning_rate_start
                    sign = 1
                    stepsize = params.t0 / 2
                    slope_counter = 0
                if (epoch + 1) % stepsize == 0:
                    sign = sign * -1
                    slope_counter = 0
                if sign == 1:
                    params.initial_training_rate = base_lr + slope_counter * (max_lr - base_lr) / stepsize
                elif sign == -1:
                    params.initial_training_rate = max_lr + slope_counter * (base_lr - max_lr) / stepsize
                slope_counter += 1

            if params.learning_rate_warmup == "True":
                # overwrite initial learning rate while warmup
                if epoch < params.t0/2:
                    params.initial_training_rate = learning_rate_start/(params.t0/2)*(epoch+1)

            num_steps = (params.train_size + params.train_batch_size - 1) // params.train_batch_size

            # If we are using clusters in the train model spec, initialize the cluster centers
            # if 'cluster_centers' in train_model_spec and epoch == 9:
            #     # Initialization of centers by running kmeans
            #     cluster_centers = train_model_spec['cluster_centers']
            #     kmeans_model = KMeansClustering(num_clusters=params.k, use_mini_batch=True, distance_metric='cosine')
            #     for i in range(20):
            #         sess.run(train_model_spec['iterator_init_op'])
            #         kmeans_model.train(
            #             lambda: sess.run(train_model_spec['samples'],
            #                              feed_dict={train_model_spec['sigma_placeholder']:
            #                                             params.sigma}), steps=num_steps)
            #     sess.run(cluster_centers.assign(kmeans_model.cluster_centers()))

            # Run one epoch
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

                if params.visualize == 1:
                    log_dir = train_writer.get_logdir()
                    metadata = os.path.join(log_dir, ('metadata' + str(epoch + 1) + '.tsv'))

                    # def save_metadata(file): necessary for projectors in tensorboard
                    with open(metadata, 'w') as metadata_file:
                        for c in embedded_labels:
                            metadata_file.write('{}\n'.format(c))
                    visualize_embeddings(sess, log_dir, embedded_data, (epoch + 1), train_writer, params)

            # If best_eval, best_save_path
            metrics_eval['training_loss'] = metrics_train['loss']

            if metrics_eval['Normalized Mutual Information'] >= best_eval_nmi:
                # Store new best accuracy
                best_eval_nmi = metrics_eval['Normalized Mutual Information']
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
