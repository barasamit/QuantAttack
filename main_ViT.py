#######shoutcut########
bitsandbytes_dir = "/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/bitsandbytes/autograd/_functions.py"
transformers_dir = "/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/transformers/bitsandbytes/"
#######################
input_arr = []
outliers_arr = []


def hook_fn(module, input, output):
    input_arr.append(input[0])
