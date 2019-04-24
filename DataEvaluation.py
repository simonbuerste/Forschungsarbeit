import os
import json
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib2tikz import save as savetikz

directory = "C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Models/LatentSpace"

list_dir = next(os.walk(directory))[1]
summary = []
visu_data = []
datasets = ["MNIST", "F-MNIST", "CIFAR-10", "CIFAR-100", "IMAGENET-Dog", "IMAGENET-10"]
model_visu = ["AE", "VAE"] # "AE", "VAE", "discriminative_AE", "featureselective_AE", "Gumbel_VAE"

metric_visu = ["Accuracy", "NMI", "ARI"]  # "Accuracy_best", "NMI_best", "ARI_best"

model_color = {
    "AE":                   'red',
    "VAE":                  'blue',
    "discriminative_AE":    'green',
    "Gumbel_VAE":           'black'}

hyperparameters = ["n_latent"] # ["n_latent", "temperature_gumbel"]  ["alpha", "lambda_r"]
for dataset_visu in datasets:
    for x in list_dir:
        # define the filenames
        filename_param = os.path.join(directory, x, "params.json")
        filename_best_weigths = os.path.join(directory, x, "metrics_eval_best_weights.json")
        filename_last_weigths = os.path.join(directory, x, "metrics_eval_last_weights.json")

        # open and read the corresponding json files
        with open(filename_param, "r") as paramsFile:
            params = json.load(paramsFile)

        # Check if results were saved
        if os.path.isfile(filename_best_weigths):
            with open(filename_best_weigths, "r") as eval_best_File:
                eval_best_weights = json.load(eval_best_File)

            with open(filename_last_weigths, "r") as eval_last_File:
                eval_last_weights = json.load(eval_last_File)

            # Check the dataset which was evaluated
            if "MNIST" in x:
                if "F-MNIST" in x:
                    dataset = "F-MNIST"
                else:
                    dataset = "MNIST"
            elif "CIFAR-10" in x:
                if "CIFAR-100" in x:
                    dataset = "CIFAR-100"
                else:
                    dataset = "CIFAR-10"
            elif "IMAGENET-Dog" in x:
                dataset = "IMAGENET-Dog"
            elif "IMAGENET-10" in x:
                dataset = "IMAGENET-10"


            # Check the Latent Model which was used
            if "AE" in x:
                if "b_AE" in x:
                    latent_model = "discriminative_AE"
                elif "fs_AE" in x:
                    latent_model = "featureselective_AE"
                elif "VAE" in x:
                    if "b_VAE" in x:
                        latent_model = "Beta_VAE"
                    elif "g_VAE" in x:
                        latent_model = "Gumbel_VAE"
                    else:
                        latent_model = "VAE"
                else:
                    latent_model = "AE"

            # Check the CLuster model which was used
            if "kmeans" in x:
                cluster_model = "kmeans"
            elif "gmm" in x:
                cluster_model = "gmm"
            elif "IDEC" in x:
                cluster_model = "IDEC"

            # Add general metrics/values/strings to dict
            tmp = {
                "Output_Folder":            x,
                "dataset":                  dataset,
                "latent_model":             latent_model,
                "cluster_model":            cluster_model,
                "test_loss":                eval_best_weights["loss"],
                "test_loss_last":           eval_last_weights["loss"],
                "training_loss":            eval_best_weights["training_loss"],
                "training_loss_last":       eval_last_weights["training_loss"],
                "test_log_likelihood":      eval_best_weights["neg_log_likelihood"],
                "test_log_likelihood_last": eval_last_weights["neg_log_likelihood"],
                "Accuracy":                 eval_best_weights["Accuracy"],
                "Accuracy_last":            eval_last_weights["Accuracy"],
                "NMI":                      eval_best_weights["Normalized Mutual Information"],
                "NMI_last":                 eval_last_weights["Normalized Mutual Information"],
                "ARI":                      eval_best_weights["Adjusted Rand Index"],
                "ARI_last":                 eval_last_weights["Adjusted Rand Index"],
            }
            # add specific hyperparameters to dict (Hyperparams defined at beginning of file)
            for i in hyperparameters:
                tmp[i] = params[i]

            summary.append(tmp)
            # Save desired data for visualization
            if (tmp["dataset"] == dataset_visu) and (tmp["latent_model"] in model_visu):
                visu_data.append(tmp)


    # Write summary dict to csv file
    keys = summary[0].keys()
    eval_file = os.path.join(directory, 'Evaluation.csv')
    with open(eval_file, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(summary)

    ##############################################################################
    # Visualization of data
    plot_metric = []
    plot_params = []

    metric_tmp = np.zeros((len(model_visu), len(visu_data), len(metric_visu)), dtype=np.float32)
    params_tmp = np.zeros((len(model_visu), len(visu_data), len(hyperparameters)), dtype=np.float32)

    iterator = np.zeros(len(model_visu), dtype=np.int64)
    for _, dict in enumerate(visu_data):
        if dict["dataset"] == dataset_visu:
            for l, model in enumerate(model_visu):
                if model == dict["latent_model"]:
                    break

            i = iterator[l]
            for j, metric in enumerate(metric_visu):
                metric_tmp[l][i][j] = dict[metric]
            for k, param in enumerate(hyperparameters):
                if dict[param] == "step_decay":
                    dict[param] = 0
                elif dict[param] == "exponential_decay":
                    dict[param] = 1
                elif dict[param] == "triangular":
                    dict[param] = 2
                elif dict[param] == "":
                    dict[param] = 3
                elif dict[param] is True:
                    dict[param] = 1
                elif dict[param] is False:
                    dict[param] = 2
                params_tmp[l][i][k] = dict[param]
            iterator[l] += 1

    for l, _ in enumerate(model_visu):
        tmp1 = metric_tmp[l, :, :]
        tmp2 = params_tmp[l, :, :]
        plot_metric.append(tmp1[~np.all(tmp1 == 0, axis=1)])
        plot_params.append(tmp2[~np.all(tmp2 == 0, axis=1)])

    if len(hyperparameters) == 1:
        for i, metric in enumerate(metric_visu):
            model_string = ""
            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            for l, model in enumerate(model_visu):
                ax.plot(plot_params[l][:, 0], plot_metric[l][:, i], c=model_color[model], linestyle='-', label=model) # model_color[model],
                if l == 0:
                    model_string = model
                else:
                    model_string = model_string + ' ' + model
                mean = np.mean(plot_metric[l][:, i])
                std = np.sqrt(np.sum((plot_metric[l][:, i]-mean)**2)/(len(plot_metric[l][:, i])-1))
                print('Mean %s of %s at %s: %.4f' % (metric, model, dataset_visu, mean))
                print('Std %s of %s at %s: %.4f' % (metric, model, dataset_visu, std))
            ax.legend()
            ax.set_xlabel('%s' % hyperparameters[0])
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            ax.set_ylabel('%s' % metric)
            ax.set_title('%s of %s at %s' % (metric, model_string, dataset_visu))
            savepath = os.path.join(directory, '%s_%s_%s_%s' % (dataset_visu, metric, model_string, hyperparameters[0]))
            savetikz(savepath + '.tex')
            plt.savefig(savepath + '.png')
            plt.savefig(savepath+'.eps', format='eps', dpi=1000)
        plt.show(block=True)
    elif len(hyperparameters) == 2:
        for l, model in enumerate(model_visu):
            for i, metric in enumerate(metric_visu):
                param_values_0 = np.unique(plot_params[l][:, 0])
                param_values_1 = np.unique(plot_params[l][:, 1])
                confusion_matrix = np.zeros((len(param_values_0), len(param_values_1)))
                for idx, elems in enumerate(plot_metric[l][:, i]):
                    confusion_idx0 = np.where(param_values_0 == plot_params[l][idx, 0])
                    confusion_idx1 = np.where(param_values_1 == plot_params[l][idx, 1])
                    confusion_matrix[confusion_idx0, confusion_idx1] = elems

                # ----- Confusion Matrix Plot -----
                fig, ax = plt.subplots()
                im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                ax.set(xticks=np.arange(confusion_matrix.shape[1]),
                       yticks=np.arange(confusion_matrix.shape[0]),
                       xticklabels=["step_decay", "exponential_decay", "triangular", "constant"], yticklabels=["true", "false"],
                       title='%s of %s at %s' % (metric, model, dataset_visu),
                       xlabel='%s' % hyperparameters[1], ylabel='%s' % hyperparameters[0])

                # Loop over data dimensions and create text annotations.
                fmt = '.2f'
                thresh = confusion_matrix.mean()
                for k in range(confusion_matrix.shape[0]):
                    for j in range(confusion_matrix.shape[1]):
                        ax.text(j, k, format(confusion_matrix[k, j], fmt),
                                ha="center", va="center",
                                color="white" if confusion_matrix[k, j] > thresh else "black")
                fig.tight_layout()
                # ----- Scatter Plot with metric values as colormap -----
                # fig = plt.figure(i)
                # ax = fig.add_subplot(111)
                # sc = ax.scatter(plot_params[l][:, 0], np.log10(plot_params[l][:, 1]), c=plot_metric[l][:, i])
                # plt.colorbar(sc)#, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                # max_metric = max(plot_metric[l][:, i])
                # idx = np.where(plot_metric[l][:, i] == max_metric)
                # xpos = plot_params[l][idx, 0]
                # ypos = np.log10(plot_params[l][idx, 1])
                # ax.set_xlabel('%s' % hyperparameters[0])
                # ax.set_ylabel('log_10 of %s' % hyperparameters[1])
                # ax.set_title('%s for different Hyperparameters for %s' % (metric, model))

                savepath = os.path.join(directory, '%s_%s_%s_%s_%s' % (dataset_visu, model, metric, hyperparameters[0], hyperparameters[1]))
                savetikz(savepath+'.tex')
                plt.savefig(savepath+'.png')
                plt.savefig(savepath+'.eps', format = 'eps', dpi = 1000)
        plt.show()
    else:
        print("Sure you want to visualize this?")
