import torch
from PIL import Image

from configs.attacks_config import config_dict
from utils.model_utils import get_model, get_vit_feature_extractor
from utils.general import get_instance

from losses import Loss


class SimpleAttack:
    def __init__(self, cfg) -> None:
        super().__init__()
        self.batch_info = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.model_name = cfg['model_name']

        self.model = get_model(cfg, self.model_name)
        self.feature_extractor = get_vit_feature_extractor()

        self.loss_func = get_instance(self.cfg['losses_config']['module_name'],
                                      self.cfg['losses_config']['class_name'])(**self.cfg['loss_params'])
        self.convert_fn = get_instance(self.cfg['model_config']['model_path'].replace('/', '.') + '.utils',
                                       'convert_to_save_format')
        self.loss = Loss(self.model, self.loss_func, self.convert_fn)
        self.cfg['attack_params']['loss_function'] = self.loss.loss_gradient
        self.attack = get_instance(self.cfg['attack_config']['module_name'],
                                   self.cfg['attack_config']['class_name'])(**self.cfg['attack_params'])

        self.attack_image_diff = self.feature_extractor(images=Image.open(cfg['attack_img_diff_path']),
                                                        return_tensors="pt")["pixel_values"]

        target_image_orig = self.feature_extractor(images=Image.open(cfg['target_img_orig_path']), return_tensors="pt")[
            "pixel_values"]
        with torch.no_grad():
            self.pred_target = self.model(target_image_orig)

    def generate(self):
        self.batch_info = {'cur': 0, 'total': 32}
        self.attack.generate(self.attack_image_diff, self.pred_target, self.batch_info)


def main():
    config_type = 'OneToOne'
    cfg = config_dict[config_type]()
    attack = SimpleAttack(cfg)
    attack.generate()


if __name__ == '__main__':
    main()
