import pandas as pd
import torch

from configs.attacks_config import config_dict
from tqdm import tqdm
from attack import Attack
import os
import argparse

def denormalize(x, mean=None, std=None):
    # 3, H, W, B
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)
#
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--num',default=1,type=int, help='Number Path to the data directory')
args = parser.parse_args()

path_dict = {
    1: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/VIT_Quant_ensemble/experiments/September/inf_0.06274509803921569_0.002_True_3000_4_1_VIT_[[1, 47.0, 0]]_70/",

}

results_combine = pd.DataFrame()
config_type = 'ManyToMany'
cfg = config_dict[config_type]()

attack = Attack(cfg)
model_name = cfg.model_name
csv_name = f"single_{model_name}.csv"
path =  path_dict[args.num]        #path_dict[args.num]
main_dir = path.rsplit("/", 2)[0]
if os.path.exists(os.path.join(path, csv_name)):
    results_combine = pd.read_csv(os.path.join(path, csv_name))
    results_combine = results_combine.drop(results_combine.index[-1])
# Lists to store the filenames
clean_files = []
adv_files = []

for filename in os.listdir(path):
    if filename.endswith('clean.pt'):
        clean_files.append(os.path.join(path, filename))
    elif filename.endswith('adv.pt'):
        adv_files.append(os.path.join(path, filename))

for i, file in enumerate(tqdm(adv_files)):
    if i < len(results_combine):  # Skip already evaluated files
        continue
    try:
        file_clean = torch.load(clean_files[i])
    except:
        file_clean = torch.load(clean_files[0])

    file_adv = torch.tensor(torch.load(adv_files[i]))
    if model_name == "Owldetection":
        ids = torch.randint(0, 100, (1, 16))
        res = attack.compute_success(file_clean, file_adv, i, "", False, ids = ids)
    else:

        res = attack.compute_success(file_clean, file_adv, i, "")
    if res["clean_CPU_time"][0] == 0:
        break
    results_combine = pd.concat([results_combine, res], axis=0)
    results_combine.to_csv(os.path.join(path, csv_name), index=False)


