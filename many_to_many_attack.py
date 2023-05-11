import math
import os
import warnings

import yaml

from configs.attacks_config import config_dict
from utils.data_utils import get_loaders
# from visualization.plots import loss_plot, cm_plot
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
        _, _, self.test_loader = get_loaders(self.cfg.loader_params, self.cfg.dataset_config, ['test'],
                                             **dataset_params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attack_parmas = dict(list(cfg.attack_params.items())[:-1])
        self.file_name = self.file_name + "_iter_" + str(self.attack_parmas['max_iter']) + ".csv"

    def generate(self, max_batch=math.inf):
        print("Starting many-to-many attack...")
        results_combine = pd.DataFrame()

        for batch_id, data in enumerate(self.test_loader):

            if batch_id > max_batch or (batch_id % 5 == 0 and batch_id > 0):
                # save results
                results_combine.to_csv(self.file_name, index=False)
                # save adv images
                save_image(adv_x[0], os.path.join(self.attack_dir, "adv.jpg"))
                save_image(attack_images[0], os.path.join(self.attack_dir, "clean.jpg"))

                # save attack parameters
                with open(os.path.join(self.attack_dir, "attack_parameters.yml"), 'w') as outfile:
                    yaml.dump(self.attack_parmas, outfile, default_flow_style=False)
                print(f"saved{self.file_name}")
            if batch_id > max_batch:
                break

            # attack
            attack_images = data[0].squeeze(1).to(self.device)
            adv_x = self.attack.generate(attack_images, attack_images,
                                         {'cur': batch_id + 1, 'total': len(self.test_loader)})

            res = self.compute_success(attack_images, adv_x, batch_id, data[1])
            results_combine = pd.concat([results_combine, res], axis=0)  # combine results


def main():
    config_type = 'ManyToMany'
    cfg = config_dict[config_type]()
    attack = ManyToManyAttack(cfg)
    attack.generate(5)  # generate k batches

    # create grid search for attack parameters
    # config_type = 'ManyToMany'
    # cfg = config_dict[config_type]()
    # max_iter_list = [500,1500,3000,4000,5000,6000]
    # for max_iter in max_iter_list:
    #     attack_params = {
    #         'norm': 2,
    #         'eps': 500,
    #         'eps_step': 1,
    #         'decay': None,
    #         'max_iter': max_iter,
    #         'targeted': True,
    #         'num_random_init': 1,
    #         'device': "cuda"
    #     }
    #     for k, v in attack_params.items():
    #         print(k, v)
    #     cfg.attack_params = attack_params
    #     cfg.max_iter = max_iter
    #     attack = ManyToManyAttack(cfg)
    #     attack.generate(5)  # generate 100 batches


def main_iter_2():
    norm_list = [2]
    eps_list = [1, 2, 5, 150, 500]
    eps_step_list = [0.05, 0.1, 1, 3, 5]
    targeted_list = [True]
    max_iter_list = [100, 300, 1000]
    num_topk_values_list = [1, 3, 5, 5000]
    choice_list = [0, 1, 2]

    # create grid search for attack parameters
    for norm, eps, eps_step, targeted, max_iter, num_topk_values in product(norm_list, eps_list, eps_step_list,
                                                                            targeted_list, max_iter_list,
                                                                            num_topk_values_list):
        attack_params = {
            'norm': norm,
            'eps': eps,
            'eps_step': eps_step,
            'decay': None,
            'max_iter': max_iter,
            'targeted': targeted,
            'num_random_init': 1,
            'device': "cuda"
        }
        for k, v in attack_params.items():
            print(k, v)

        config_type = 'ManyToMany'
        cfg = config_dict[config_type]()
        cfg.attack_params = attack_params
        cfg.num_topk_values = num_topk_values
        cfg.choice = 0

        attack = ManyToManyAttack(cfg)
        # if os.path.exists(attack.attack_dir):
        #     continue

        attack.generate(100)  # generate 100 batches
        print("#############################################")


def main_iter_inf():
    norm_list = ["inf"]
    eps_list = [10]
    eps_step_list = [0.01]
    max_iter_list = [1000]
    num_topk_values_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    targeted_list = [True]

    # create grid search for attack parameters
    for norm, eps, eps_step, targeted, max_iter, num_topk_values in product(norm_list, eps_list, eps_step_list,
                                                                            targeted_list, max_iter_list,
                                                                            num_topk_values_list):
        attack_params = {
            'norm': norm,
            'eps': eps,
            'eps_step': eps_step,
            'decay': None,
            'max_iter': max_iter,
            'targeted': targeted,
            'num_random_init': 1,
            'device': "cuda"
        }
        for k, v in attack_params.items():
            print(k, v)

        config_type = 'ManyToMany'
        cfg = config_dict[config_type]()
        cfg.attack_params = attack_params
        cfg.num_topk_values = 1
        cfg.choice = 0

        attack = ManyToManyAttack(cfg)
        # if os.path.exists(attack.attack_dir): continue
        #
        attack.generate(10)  # generate 100 batches
        print("#############################################")


if __name__ == '__main__':
    main()
