import torch
from PIL import Image
import matplotlib.pyplot as plt

from configs.attacks_config import config_dict
from utils.general import load_npy_image_to_tensor
from attack import Attack
# from visualization.plots import loss_plot
import seaborn as sns
import pandas as pd


class OneToOneAttack(Attack):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.attack_image_diff = self.feature_extractor(images=Image.open(cfg['attack_img_diff_path']),
                                                        return_tensors="pt")["pixel_values"]
        self.pred_target = self.feature_extractor(images=Image.open(cfg['target_img_orig_path']), return_tensors="pt")[
            "pixel_values"]

    def generate(self):
        _ = self.attack.generate(self.attack_image_diff.to("cuda"), self.pred_target, {'cur': 1, 'total': 1})



def main():
    config_type = 'OneToOne'
    cfg = config_dict[config_type]()
    attack = OneToOneAttack(cfg)
    attack.generate()


if __name__ == '__main__':
    main()