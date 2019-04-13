import os
import json
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib2tikz import save as savetikz

directory = "C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Models/InitLearningRate"

list_dir = next(os.walk(directory))[1]
summary = []
visu_data = []
dataset_visu = ["MNIST"]
model_visu = ["AE", "VAE"] # "AE", "VAE", "discriminative_AE", "featureselective_AE", "Gumbel_VAE"

hyperparameters = ["train_batch_size", "initial_training_rate"]
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
            "test_loss_best":           eval_best_weights["loss"],
            "test_loss_last":           eval_last_weights["loss"],
            "training_loss_best":       eval_best_weights["training_loss"],
            "training_loss_last":       eval_last_weights["training_loss"],
            "test_log_likelihood_best": eval_best_weights["neg_log_likelihood"],
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
        if (tmp["dataset"] in dataset_visu) and (tmp["latent_model"] in model_visu):
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
metric_visu = ["Accuracy", "NMI"] # "Accuracy_best", "NMI_best", "ARI_best"

model_color = {
    "AE":                   'red',
    "VAE":                  'blue',
    "discriminative_AE":    'green',
    "Gumbel_VAE":           'black'
}

plot_metric = np.zeros((len(visu_data), len(model_visu), len(metric_visu)), dtype=np.float32)
plot_params = np.zeros((len(visu_data), len(model_visu), len(hyperparameters)), dtype=np.float32)

iterator = np.zeros(len(model_visu), dtype=np.int64)
for _, dict in enumerate(visu_data):
    for l, model in enumerate(model_visu):
        if model == dict["latent_model"]:
            break

    i = iterator[l]
    for j, metric in enumerate(metric_visu):
        plot_metric[i][l][j] = dict[metric]
    for k, param in enumerate(hyperparameters):
        plot_params[i][l][k] = dict[param]
    iterator[l] += 1

if len(hyperparameters) == 1:
    for i, metric in enumerate(metric_visu):
        fig = plt.figure(i)
        ax = fig.add_subplot(111)
        for l, model in enumerate(model_visu):
            ax.plot(plot_params[:, l, 0], plot_metric[:, l, i], c=model_color[model], linestyle='-', label=model) # model_color[model],
        ax.legend()
        ax.set_xlabel('%s' % hyperparameters[0])
        ax.set_ylim(0, 1)
        ax.set_ylabel('%s' % metric)
        ax.set_title('%s for different %s' % (metric, hyperparameters[0]))
        savepath = os.path.join(directory, '%s.tex' % metric)
        savetikz(savepath)
    plt.show(block=True)
elif len(hyperparameters) == 2:
    for l, model in enumerate(model_visu):
        for i, metric in enumerate(metric_visu):
            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            sc = ax.scatter(plot_params[:, l, 0], np.log10(plot_params[:, l, 1]), c=plot_metric[:, l, i])
            plt.colorbar(sc)#, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            max_metric = max(plot_metric[:, l, i])
            idx = np.where(plot_metric[:, l, i] == max_metric)
            xpos = plot_params[idx, l, 0]
            ypos = np.log10(plot_params[idx, l, 1])
            #ax.text(xpos, ypos, 'Maximum: %.3f' % max_metric)
            #arrowprops = {"facecolor": 'black', "shrink": 0.05}
            #ax.annotate('maximum Value: %.3f' % max_metric, xy=(xpos, ypos), xytext=(xpos, ypos-0.5))#, arrowprops=arrowprops)
            ax.set_xlabel('%s' % hyperparameters[0])
            ax.set_ylabel('log_10 of %s' % hyperparameters[1])
            ax.set_title('%s for different Hyperparameters for %s' % (metric, model))
            savepath = os.path.join(directory, '%s.tex' % metric)
            savetikz(savepath)
        plt.show()
else:
    print("Sure you want to visualize this?")