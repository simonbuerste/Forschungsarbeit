import os
import subprocess
import time
import json

n_latent = [10, 32, 64, 128, 256]
filter_first_layer = [8, 16, 32, 64, 128] # 64 already checked

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

for n in filter_first_layer:
    # part for changing the parameters as desired

    # first read the params and overwrite the specific parameter
    with open("params.json", "r") as paramsFile:
        params = json.load(paramsFile)
    params["filter_first_layer"] = n
    # write the new parameters to the parameter file
    with open("params.json", "w") as paramsFile:
        json.dump(params, paramsFile)
    # execute the Training for all Datasets
    #p = subprocess.Popen('python3.6 Train.py --dataset=MNIST --gpu=2 --latent_model=AE')
    #p.wait()
    os.system('python3.6 Train.py --dataset=MNIST --gpu=2 --latent_model=AE')
    time.sleep(70)  # 70 seconds pause to ensure models are not written in same folder
