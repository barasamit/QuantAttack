import argparse
import math
import os

import yaml

from configs.attacks_config import config_dict
from utils.data_utils import get_loaders
from attack import Attack
import pandas as pd
import torch
from itertools import product
from torchvision.utils import save_image
from tqdm import tqdm


#####
class ManyToManyAttack(Attack):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        dataset_params = {'random_pair': True, 'load_reconstruct': True, 'load_lensed': True}
        _, self.test_loader, _ = get_loaders(self.cfg.loader_params, self.cfg.dataset_config, ['validation', 'test'],
                                             model_name=self.cfg.model_name,
                                             **dataset_params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attack_parmas = dict(list(cfg.attack_params.items())[:-1])
        self.file_name = self.file_name + "_iter_" + str(self.attack_parmas['max_iter']) + ".csv"

    def generate(self, max_batch=math.inf):
        print("Starting many-to-many attack...")
        results_combine = pd.DataFrame()

        for batch_id, data in enumerate(self.test_loader):

            if batch_id > max_batch or (batch_id % 1 == 0 and batch_id > 0):
                # save results
                results_combine.to_csv(self.file_name, index=False)
                # save adv images
                save_image(self.denormalize(adv_x,self.model_mean,self.model_std), os.path.join(self.attack_dir, "adv.jpg"))
                torch.save(adv_x, os.path.join(self.attack_dir, "adv.pt"))
                save_image(self.denormalize(attack_images,self.model_mean,self.model_std), os.path.join(self.attack_dir, "clean.jpg"))
                torch.save(attack_images, os.path.join(self.attack_dir, "clean.pt"))
                torch.save(adv_x[0] - attack_images[0], os.path.join(self.attack_dir, "perturbation_torch.pt"))

                # save attack parameters
                with open(os.path.join(self.attack_dir, "attack_parameters.yml"), 'w') as outfile:
                    yaml.dump(self.attack_parmas, outfile, default_flow_style=False)
                print(f"saved{self.file_name}")
            if batch_id > max_batch:
                break

            # attack
            attack_images = data[0].squeeze(1).to(self.device)
            adv_x = self.attack.generate(attack_images, attack_images,
                                         {'cur': batch_id + 1, 'total': len(self.test_loader)}, data[-1])

            res = self.compute_success(attack_images, adv_x, batch_id, data[1], None, data[-1])
            results_combine = pd.concat([results_combine, res], axis=0)  # combine results

            if batch_id > 500:
                break

    def denormalize(self, x, mean=None, std=None):
        # 3, H, W, B
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        ten = x.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def parse_args():
    parser = argparse.ArgumentParser(description='Many-to-Many Attack')
    parser.add_argument('--accuracy_loss', type=float, default=50, help='Weight for accuracy loss')
    parser.add_argument('--TV_loss', type=float, default=0.01, help='Weight for Total Variation loss')
    return parser.parse_args()


def main():
    args = parse_args()

    config_type = 'ManyToMany'
    cfg = config_dict[config_type]()

    accuracy_loss = args.accuracy_loss
    TV_loss = args.TV_loss

    cfg.loss_params = {'weights': [[1, accuracy_loss,
                                   TV_loss]]}  # 0 - loss, 1 - loss on the accuracy, 2 - loss on the total variation [1, 50, 0.01]

    attack = ManyToManyAttack(cfg)
    attack.generate(1000)  # generate k batches



# def main_iter_2():
#     norm_list = [2]
#     eps_list = [500]
#     eps_step_list = [0.05, 0.1, 1, 3, 5]
#     targeted_list = [True]
#     max_iter_list = [1200]
#     num_topk_values_list = [4]
#     choice_list = [0, 1, 2]
#
#     # create grid search for attack parameters
#     for norm, eps, eps_step, targeted, max_iter, num_topk_values in product(norm_list, eps_list, eps_step_list,
#                                                                             targeted_list, max_iter_list,
#                                                                             num_topk_values_list):
#         attack_params = {
#             'norm': norm,
#             'eps': eps,
#             'eps_step': eps_step,
#             'decay': None,
#             'max_iter': max_iter,
#             'targeted': targeted,
#             'num_random_init': 1,
#             'device': "cuda"
#         }
#         for k, v in attack_params.items():
#             print(k, v)
#
#         config_type = 'ManyToMany'
#         cfg = config_dict[config_type]()
#         cfg.attack_params = attack_params
#         cfg.num_topk_values = num_topk_values
#         cfg.choice = 0
#
#         attack = ManyToManyAttack(cfg)
#         # if os.path.exists(attack.attack_dir):
#         #     continue
#
#         attack.generate(5)  # generate 100 batches
#         print("#############################################")


# def main_iter_inf():
#     norm_list = ["inf"]
#     eps_list = [10]
#     eps_step_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
#     max_iter_list = [1200]
#     num_topk_values_list = [4]
#     targeted_list = [True]
#
#     # create grid search for attack parameters
#     for norm, eps, eps_step, targeted, max_iter, num_topk_values in product(norm_list, eps_list, eps_step_list,
#                                                                             targeted_list, max_iter_list,
#                                                                             num_topk_values_list):
#         attack_params = {
#             'norm': norm,
#             'eps': eps,
#             'eps_step': eps_step,
#             'decay': None,
#             'max_iter': max_iter,
#             'targeted': targeted,
#             'num_random_init': 1,
#             'device': "cuda"
#         }
#         for k, v in attack_params.items():
#             print(k, v)
#
#         config_type = 'ManyToMany'
#         cfg = config_dict[config_type]()
#         cfg.attack_params = attack_params
#         cfg.num_topk_values = 4
#         cfg.choice = 0
#
#         attack = ManyToManyAttack(cfg)
#         # if os.path.exists(attack.attack_dir): continue
#         #
#         attack.generate(5)  # generate 100 batches
#         print("#############################################")
#

if __name__ == '__main__':
    main()
