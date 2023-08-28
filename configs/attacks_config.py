import os
import yaml
import sys
from pathlib import Path
import torch
import datetime
import time

from utils.general import init_seeds

init_seeds()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class BaseConfig:
    def __init__(self):
        self.root_path = ROOT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_name = 'VIT'  # DeiT, VIT, Whisper, Owldetection , other
        attack_type = 'universal'  # 'universal', 'single', 'class'  # check attack config yml

        model_config = {'VIT': 0, 'DeiT': 0, 'Whisper': 1, 'Owldetection': 2, 'other': 3}
        loss_param_config = {'VIT': [1, 50, 0.01], 'DeiT': [1, 50, 0.01], 'Whisper': [1, 0, 0], 'Owldetection': [1, 0, 0], 'other': [1, 0, 0]}
        self.model_config_num = model_config[model_name]

        print("Using model: ", model_name)

        dataset_name = 'imagenet'
        self._set_dataset(dataset_name)

        self.loss_func_params = {'MSE': {}}  # BCEWithLogitsLoss , MSE
        self.attack_name = 'PGD'

        if attack_type == 'single':
            self.loss_params = {
                'weights': [loss_param_config[model_name]]
                # 0 - loss, 1 - loss on the accuracy, 2 - loss on the total variation [1, 50, 0.01]
            }
        else:
            self.loss_params = {
                'weights': [loss_param_config["other"]]}

        self.attack_params = {
            'norm': "inf",  # "inf", 2, 1
            'eps': 0.2,  # 0.3 best
            'eps_step': 0.0025,  # 0.00025 best, its also lr in universal attack
            'decay': None,
            'max_iter': 2999,
            'targeted': True,
            'num_random_init': 1,
            'device': self.device,
            'clip_values': (-3, 3),
            "normalized_std": None,  # [0.485, 0.456, 0.406] for imagenet
        }

        ##################################################################################
        # Universal attack
        self.initial_patch = "zeros"  # "random" ,"zero", "ones"
        self.image_size = 224
        self.epochs = 500
        self.number_of_training_images = 250
        self.number_of_val_images = 50
        self.number_of_test_images = 100
        ##################################################################################

        self.max_iter = self.attack_params['max_iter']
        self.use_scheduler = False
        self.scheduler = 0.5

        self.demo_mode = 'predict_one'  # 'predict_one' or 'predict_many'

        self.loader_params = {
            'batch_size': 1,
            'num_workers': 1
        }

        self.model_threshold = 6
        self.model_threshold_dest = 7
        self.bottom_threshold = 2
        self.target = 70

        self.num_topk_values = 4
        print("num_topk_values: ", self.num_topk_values)
        self.choice = 0  # 0: topk for each column for each layer, 1: top k from small layers(782)->In reference to all without division , 2: same as 1 but also for large layers(3072)

        self.blocks_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 12 blocks in VIT

        self._set_model(model_name)
        self._set_losses(self.loss_func_params)

        with open(ROOT / 'configs/attack_config.yaml', 'r') as stream:
            self.attack_config = yaml.safe_load(stream)[self.attack_name]

        self._update_current_dir()

    def __getitem__(self, item):
        return getattr(self, item)

    def _set_model(self, model_name):
        self.model_name = model_name
        with open(ROOT / 'configs/model_config.yaml', 'r') as stream:
            self.model_config = yaml.safe_load(stream)[self.model_name]

    def _set_losses(self, loss_func_params):
        self.loss_func_params = loss_func_params
        with open(ROOT / 'configs/losses_config.yaml', 'r') as stream:
            yaml_file = yaml.safe_load(stream)
            self.losses_config = []
            for loss_name in self.loss_func_params.keys():
                self.losses_config = yaml_file[loss_name]
            del yaml_file

    def _update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = os.path.join("experiments", month_name)
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)

    def _set_dataset(self, dataset_name):
        with open(ROOT / 'configs/dataset_config.yaml', 'r') as stream:
            self.dataset_config = yaml.safe_load(stream)[dataset_name]
        self.dataset_config['dataset_name'] = dataset_name


class OneToOneAttackConfig(BaseConfig):
    def __init__(self):
        super(OneToOneAttackConfig, self).__init__()
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_path = os.path.join(self.root_path, 'demo')

        attack_image_name = 'egyptian_cat.jpg'  # egyptian_cat.jpg
        self.attack_img_diff_path = os.path.join(dataset_path, attack_image_name)
        target_image_name = 'corgi.jpg'
        self.target_img_orig_path = os.path.join(dataset_path, target_image_name)
        # self.loss_params.update({'images_save_path': os.path.join(self.current_dir, 'outputs')})


class ManyToManyAttackConfig(BaseConfig):
    def __init__(self):
        super(ManyToManyAttackConfig, self).__init__()


class UniversalAttackConfig(BaseConfig):
    def __init__(self):
        super(UniversalAttackConfig, self).__init__()


config_dict = {
    'Base': BaseConfig,
    'OneToOne': OneToOneAttackConfig,
    'ManyToMany': ManyToManyAttackConfig,
    'Universal': UniversalAttackConfig
}
