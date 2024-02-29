import pandas as pd
import torch
from utils.init_collect_arrays import hook_fn, outliers_arr_local
from transformers import ViTForImageClassification
from utils.attack_utils import count_outliers
from configs.attacks_config import config_dict
from tqdm import tqdm
from attack import Attack
from torchvision.utils import save_image
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


path_dict = {1: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/vit/inf_0.06274509803921569_0.002_True_2999_4_1_VIT_[[1, 50.0, 0.0]]_70/",
             2: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/vit/inf_0.06274509803921569_0.002_True_2999_4_1_VIT_[[1, 50.0, 0.0]]_70/",
             3: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/ablation_study/acc/inf_0.06274509803921569_0.002_True_2998_4_1_VIT_[[1, 0.0, 0.0]]_70/",
             4: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/ablation_study/acc/inf_0.06274509803921569_0.002_True_2998_4_1_VIT_[[1, 25.0, 0.0]]_70/",
             5: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/ablation_study/acc/inf_0.06274509803921569_0.002_True_2998_4_1_VIT_[[1, 100.0, 0.0]]_70/",
             6: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/ablation_study/acc/inf_0.06274509803921569_0.002_True_2998_4_1_VIT_[[1, 75.0, 0.0]]_70/",
             7: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/ablation_study/acc/inf_0.06274509803921569_0.002_True_2998_4_1_VIT_[[1, 250.0, 0.0]]_70/",
             8: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/ablation_study/epsilons/inf_0.047058823529411764_0.002_True_2999_4_1_VIT_[[1, 50.0, 0.0]]_70/",
             9: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/ablation_study/epsilons/inf_0.12549019607843137_0.002_True_2999_4_1_VIT_[[1, 50.0, 0.0]]_70/",
             10: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/ablation_study/epsilons/inf_0.03137254901960784_0.002_True_2999_4_1_VIT_[[1, 50.0, 0.0]]_70/",
             11: "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/Final_results/baselines/Normal_PGD/normal_pgd_vit/",
             }



parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--num',default=1,type=int, help='Number Path to the data directory')
args = parser.parse_args()


results_combine = pd.DataFrame()
config_type = 'ManyToMany'
cfg = config_dict[config_type]()
attack = Attack(cfg)
model_name = cfg.model_name

csv_name = f"eval_all.csv"
path = path_dict[args.num]
main_dir = path
if os.path.exists(os.path.join(main_dir, csv_name)):
    results_combine = pd.read_csv(os.path.join(main_dir, csv_name))
    # results_combine = results_combine.drop(results_combine.index[-1])

old_columns = ['batch_id', "img_dir", "clean_power_usage", "clean_CUDA_time", "clean_CUDA_mem", "clean_outliers"]
columns = old_columns + ["random", "adv"]
# columns = old_columns + ["adv"]
universal_patch = None
# if model_name in ["VIT", "DeiT"]:
#     universal_patch = torch.load(
#         "/dt/shabtaia/dt-fujitsu/8_bit_attack/final_results/Universal/VIT/lr_0.002_epsilon_0.8_weights[[1, 50.0, 0.01]]_nameVIT.csv/perturbation_torch.pt")
#     columns = old_columns + ["random", "Universal", "adv"]

# Lists to store the filenames
clean_files = []
adv_files = []
ids = []
for filename in os.listdir(path):
    if filename.endswith('clean.pt'):
        clean_files.append(os.path.join(path, filename))
    elif filename.endswith('adv.pt'):
        adv_files.append(os.path.join(path, filename))

for i, file in enumerate(tqdm(clean_files)):
    if i < len(results_combine):  # Skip already evaluated files
        continue

    file_clean = torch.load(clean_files[i])
    file_adv = torch.load(adv_files[i])
    image_size = file_adv.size()

    max_value = file_adv.max()
    min_value = file_adv.min()

    # Generate a random image with values sampled from a normal distribution
    mean = 0.5 * (max_value + min_value)  # Set the mean to be in the middle of the range
    std_dev = (max_value - min_value) / 4  # Adjust the standard deviation as needed
    random_image = torch.randn(image_size).to("cuda") * std_dev + mean
    random_image = file_clean + random_image
    if model_name == "Owldetection":
        ids = torch.randint(0, 100, (1, 10, 16))
    if universal_patch is not None:
        universal_image = file_clean + universal_patch
        res = attack.compute_success2(file_clean, [random_image, universal_image, file_adv], i, "")

    else:
        res = attack.compute_success2(file_clean, [random_image,file_adv], i, "", ids=ids) #[random_image, file_adv]
    res.columns = columns
    results_combine = pd.concat([results_combine, res], axis=0)
    # replace columns names

    results_combine.to_csv(os.path.join(main_dir, csv_name), index=False)
    torch.cuda.empty_cache()
    # print(results_combine)
