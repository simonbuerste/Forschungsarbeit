import os
import time
import json
from shutil import copyfile

datasets = ["MNIST", "F-MNIST", "CIFAR-10", "CIFAR-100", "IMAGENET-10", "IMAGENET-Dog"] #"MNIST", "F-MNIST", "CIFAR-10", "CIFAR-100", "IMAGENET-10", "IMAGENET-Dog"] #
n_latent = [5, 10, 20, 32, 64, 128, 256] #5, 10, 20, 32, 64, 128, 256

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

filename_params = "params_VAE"
tmp_filename = "{}.json".format(filename_params)
i = 0

while os.path.exists(tmp_filename):
    tmp_filename = "{}_{}.json".format(filename_params, i)
    i += 1

filename_params = tmp_filename + ".json"

copyfile("params.json", filename_params)

for dataset in datasets:
    # adapt dataset specific parameters (i.e. train size, num_classes,...)
    # first read the params and overwrite the specific parameter
    with open(filename_params, "r") as paramsFile:
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
    with open(filename_params, "w") as paramsFile:
        json.dump(params, paramsFile)

    for n in n_latent:
        # part for changing the parameters as desired

        # first read the params and overwrite the specific parameter
        with open(filename_params, "r") as paramsFile:
            params = json.load(paramsFile)
        params["n_latent"] = n
        # write the new parameters to the parameter file
        with open(filename_params, "w") as paramsFile:
            json.dump(params, paramsFile)
        # execute the Training for all Datasets
        #p = subprocess.Popen('python3.6 Train.py --dataset=MNIST --gpu=2 --latent_model=AE')
        #p.wait()
        os.system('python3.6 Train.py --dataset=%s --gpu=3 --latent_model=VAE --cluster_model=kmeans --Parameters=%s'% (dataset, filename_params))
        time.sleep(70)  # 70 seconds pause to ensure models are not written in same folder

# remove the copied version of params file,
# since the model specific parameters are saved at model direction
os.remove(filename_params)
