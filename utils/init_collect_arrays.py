#######shoutcut########
import torch.nn
from bitsandbytes.nn import Linear8bitLt

bitsandbytes_dir = "/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/bitsandbytes/autograd/_functions.py"
transformers_dir = "/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/transformers/bitsandbytes/"
#######################
input_arr = []
outliers_arr = []
outliers_arr_local = []
all_act = []
layer_norm_arr = []

def hook_fn(module, input, output):

    if isinstance(module, Linear8bitLt):
        input_arr.append(input[0])
        outliers_arr_local.append(input[0]) # this will be used to count outliers instead of ..../site-packages/bitsandytes/autograd/_functions

    if isinstance(module, torch.nn.LayerNorm):
        # pass
        layer_norm_arr.append(input[0])

