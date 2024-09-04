#######shoutcut########
import torch.nn
from bitsandbytes.nn import Linear8bitLt
from PTQ4ViT.quant_layers.linear import PTQSLBatchingQuantLinear, PostGeluPTQSLBatchingQuantLinear,PTQSLQuantLinear, MinMaxQuantLinear,PostGeluPTQSLQuantLinear
from RepQ_ViT.classification.quant.quant_modules import QuantConv2d, QuantLinear, QuantMatMul

bitsandbytes_dir = "/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/bitsandbytes/autograd/_functions.py"
transformers_dir = "/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/transformers/bitsandbytes/"
#######################
input_arr = []
outliers_arr = []
outliers_arr_local = []
all_act = []
layer_norm_arr = []
layer_norm_module = []

def hook_fn(module, input, output):
    # all_act.append(input[0]) # for sponge

    if isinstance(module, Linear8bitLt):
        input_arr.append(input[0])
        outliers_arr_local.append(input[0]) # this will be used to count outliers instead of ..../site-packages/bitsandytes/autograd/_functions

    # for ptq4vit
    if isinstance(module, (PTQSLBatchingQuantLinear, PostGeluPTQSLBatchingQuantLinear,PTQSLQuantLinear, MinMaxQuantLinear,PostGeluPTQSLQuantLinear)):
        input_arr.append(input[0])
        outliers_arr_local.append(input[0])

    # for ReQ
    if isinstance(module, ( QuantConv2d, QuantLinear, QuantMatMul)):
        input_arr.append(input[0])
        outliers_arr_local.append(input[0])

    # enable only if no quant
    # if isinstance(module, torch.nn.Linear):
    #     input_arr.append(input[0])
    #     outliers_arr_local.append(input[0])

    # for layer norm
    # if isinstance(module, torch.nn.LayerNorm):
    #     # pass
    #     layer_norm_arr.append(input[0])
    #     # layer_norm_module.append(module)

