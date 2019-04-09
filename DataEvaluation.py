import os
import json
import csv

directory = "C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Models/LatentSpace"

list_dir = next(os.walk(directory))[1]
summary = []

hyperparameters = ["n_latent"]
for x in list_dir:
    # define the filenames
    filename_param = os.path.join(directory, x, "params.json")
    filename_best_weigths = os.path.join(directory, x, "metrics_eval_best_weights.json")
    filename_last_weigths = os.path.join(directory, x, "metrics_eval_last_weights.json")

    # open and read the corresponding json files
    with open(filename_param, "r") as paramsFile:
        params = json.load(paramsFile)
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
        "Accuracy_best":            eval_best_weights["Accuracy"],
        "Accuracy_last":            eval_last_weights["Accuracy"],
        "NMI_best":                 eval_best_weights["Normalized Mutual Information"],
        "NMI_last":                 eval_last_weights["Normalized Mutual Information"],
        "ARI_best":                 eval_best_weights["Adjusted Rand Index"],
        "ARI_last":                 eval_last_weights["Adjusted Rand Index"],
    }
    # add specific hyperparameters to dict (Hyperparams defined at beginning of file)
    for i in hyperparameters:
        tmp[i] = params[i]

    summary.append(tmp)


# Write summary dict to csv file
keys = summary[0].keys()
eval_file = os.path.join(directory, 'Evaluation.csv')
with open(eval_file, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(summary)
