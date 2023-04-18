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

        model_name = 'VIT'  # see configs/model_config.yaml for other options
        self.estimator_name = 'ResNet18'

        dataset_name = 'imagenet'
        self._set_dataset(dataset_name)

        self.loss_func_params = {'MSE': {}}
        self.loss_params = {
            'weights': [1]
        }

        self.attack_name = 'PGD'
        self.attack_params = {
            'norm': 2,
            'eps': 500,
            'eps_step': 1,
            'decay': None,
            'max_iter': 300,
            'targeted': True,
            'num_random_init': 1,
            'device': self.device
        }

        self.demo_mode = 'predict_one'  # 'predict_one' or 'predict_many'
        self.record_embedding_layer = True
        self.layers_to_record = ["down5_encode_0"]
        self.reduction_method = 'PCA'
        self.output_path_for_reconstructions = os.path.join(ROOT,"output")
        self.save_reconstructions_as_jpg = True
        self.save_reconstructions_as_np = False
        self.loader_params = {
            'batch_size': 8,
            'num_workers': 4
        }

        self._set_model(model_name)
        self._set_losses(self.loss_func_params)

        with open(ROOT / 'configs/attack_config.yaml', 'r') as stream:
            self.attack_config = yaml.safe_load(stream)[self.attack_name]
        with open(ROOT / 'configs/classifier_config.yaml', 'r') as stream:
            self.estimator_config = yaml.safe_load(stream)[self.estimator_name]
        self.estimator_config.update({'device': self.device, 'mode': 'eval'})

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

        self.loader_params = {
            'batch_size': 1,
            'num_workers': 4
        }


class UniversalAttackConfig(BaseConfig):
    def __init__(self):
        super(UniversalAttackConfig, self).__init__()


config_dict = {
    'Base': BaseConfig,
    'OneToOne': OneToOneAttackConfig,
    'ManyToMany': ManyToManyAttackConfig,
    'Universal': UniversalAttackConfig
}
