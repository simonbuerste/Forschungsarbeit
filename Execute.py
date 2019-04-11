import os
import time
import json
from shutil import copyfile


latent_model = ["AE", "VAE"] # "AE", "VAE", "g_VAE", "b_AE"
datasets = ["MNIST", "F-MNIST", "CIFAR-10", "CIFAR-100", "IMAGENET-10", "IMAGENET-Dog"] #"MNIST", "F-MNIST", "CIFAR-10", "CIFAR-100", "IMAGENET-10", "IMAGENET-Dog"] #

n_latent = 5#5, 10, 20, 32, 64, 128, 256
train_batch_size = [64, 128, 256, 512, 1024]
initial_learning_rate = [0.001, 0.01, 0.1]
#lambda_r  = [10]#[1, 0.1, 0.01, 0.001]
#alpha = [0, 0.5, 1]
#temperature_gumbel = [10, 5, 1, 0.5]

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

no_repetitions = 1


for model in latent_model:

    filename_params = "params_" + latent_model
    tmp_filename = "{}.json".format(filename_params)
    i = 0
    while os.path.exists(tmp_filename):
        tmp_filename = "{}_{}.json".format(filename_params, i)
        i += 1

    copyfile("params.json", tmp_filename)

    for dataset in datasets:
        # adapt dataset specific parameters (i.e. train size, num_classes,...)
        # first read the params and overwrite the specific parameter
        with open(tmp_filename, "r") as paramsFile:
            params = json.load(paramsFile)

        if dataset == "MNIST":
            params["train_size"] = 55000
            params["eval_size"] = 10000
            params["k"] = 10
            params["resize_height"] = 32
            params["resize_width"] = 32
        elif dataset == "F-MNIST" or dataset == "CIFAR-10":
            params["train_size"] = 50000
            params["eval_size"] = 10000
            params["k"] = 10
            params["resize_height"] = 32
            params["resize_width"] = 32
        elif dataset == "CIFAR-100":
            params["train_size"] = 50000
            params["eval_size"] = 10000
            params["k"] = 20
            params["resize_height"] = 32
            params["resize_width"] = 32
        elif dataset == "IMAGENET-10":
            params["train_size"] = 13000
            params["eval_size"] = 500
            params["k"] = 10
            params["resize_height"] = 64
            params["resize_width"] = 64
        elif dataset == "IMAGENET-Dog":
            params["train_size"] = 19472
            params["eval_size"] = 750
            params["k"] = 15
            params["resize_height"] = 64
            params["resize_width"] = 64

        # write the new parameters to the parameter file
        with open(tmp_filename, "w") as paramsFile:
            json.dump(params, paramsFile)

        for n in train_batch_size:
            for k in initial_learning_rate:
                # first read the params and overwrite the specific parameter
                with open(tmp_filename, "r") as paramsFile:
                    params = json.load(paramsFile)

                # part for changing the parameters as desired
                params["train_batch_size"] = n
                params["initial_training_rate"] = k

                params["n_latent"] = n_latent

                # Gumbel VAE
                #params["temperature_gumbel"] = k

                # Discriminative AE
                #params["lambda_d"] = 1.0
                #params["lambda_r"] = n
                #params["alpha"] = k

                # write the new parameters to the parameter file
                with open(tmp_filename, "w") as paramsFile:
                    json.dump(params, paramsFile)

                # execute the Training for all Datasets
                for i in range(no_repetitions):
                    os.system('python3.6 Train.py --dataset=%s --gpu=1 --latent_model=%s --cluster_model=kmeans --Parameters=%s'% (dataset, model, tmp_filename))
                    time.sleep(70)  # 70 seconds pause to ensure models are not written in same folder

# remove the copied version of params file,
# since the model specific parameters are saved at model direction
os.remove(tmp_filename)
