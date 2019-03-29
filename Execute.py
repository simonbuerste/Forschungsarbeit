import os
import json

n_latent = [10, 32, 64, 128, 256]

for n in n_latent:
    # part for changing the parameters as desired

    # first read the params and overwrite the specific parameter
    with open("params.json", "r") as paramsFile:
        params = json.load(paramsFile)
    params["n_latent"] = n
    # write the new parameters to the parameter file
    with open("params.json", "w") as paramsFile:
        json.dump(params, paramsFile)
    # execute the Training for all Datasets
    os.system("python3.6 Train.py ----dataset=MNIST")