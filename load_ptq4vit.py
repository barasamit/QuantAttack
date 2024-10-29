
from importlib import reload,import_module
import os
import PTQ4ViT.utils.datasets as datasets
import PTQ4ViT.utils.net_wrap as net_wrap
from PTQ4ViT.utils.quant_calib import QuantCalibrator, HessianQuantCalibrator
from PTQ4ViT.utils.models import get_net
import sys
import torch
def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    target_directory = os.path.join(current_file_directory, 'PTQ4ViT', 'configs')

    _,_,files = next(os.walk(target_directory))
    if config_name + ".py" in files:
        config_directory = "/sise/home/barasa/8_bits_attack/PTQ4ViT/configs"
        config_module_name = "PTQ4ViT"

        # Add the directory to sys.path
        if config_directory not in sys.path:
            sys.path.insert(0, config_directory)

        # Import the module
        quant_cfg = import_module(config_module_name)
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg

def get_ptq4vit_net(Quant = True,m = "vit"):
    if m == "vit":
        net = 'vit_base_patch16_224'
    else:
        net = 'deit_base_patch16_224'
    config = {'config': 'PTQ4ViT', 'net': net}
    conf = config["config"]
    quant_cfg = init_config(conf)
    net_name = config['net']
    net = get_net(net_name)

    if Quant:

        wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)

        g = datasets.ViTImageNetLoaderGenerator("/dt/shabtaia/dt-fujitsu/datasets/others/imagnet/val_in_folders/", 'imagenet',
                                                32, 32, 16, kwargs={"model": net})
        calib_loader = g.calib_loader(num=32)

        quant_calibrator = HessianQuantCalibrator(net, wrapped_modules, calib_loader, sequential=False,
                                                  batch_size=4)  # 16 is too big for ViT-L-16
        # quant_calibrator.batching_quant_calib()
        # torch.save(net.state_dict(), f"/sise/home/barasa/8_bits_attack/PTQ4ViT/weights/{net_name}.pt")
        net.load_state_dict(torch.load(f"/sise/home/barasa/8_bits_attack/PTQ4ViT/weights/{net_name}.pt"))

    return net

# net = get_ptq4vit_net()