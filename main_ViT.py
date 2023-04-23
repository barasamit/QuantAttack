#######shoutcut########
import torch.nn
from bitsandbytes.nn import Linear8bitLt
bitsandbytes_dir = "/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/bitsandbytes/autograd/_functions.py"
transformers_dir = "/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/transformers/bitsandbytes/"
#######################
input_arr = []
outliers_arr = []
outliers_arr_local = []


def hook_fn(module, input, output):
    # print(module)
    # input_arr.append(input[0])
    if isinstance(module, torch.nn.LayerNorm):
        # input_arr.append(input[0])
        pass
    if isinstance(module, torch.nn.Dropout):
        pass
    if isinstance(module, Linear8bitLt):
        input_arr.append(input[0])
        outliers_arr_local.append(input[0]) # this will be used to count outliers instead of ..../site-packages/bitsandytes/autograd/_functions

    # if isinstance(module, torch.nn.LayerNorm):
    #     input_arr.append(output)

