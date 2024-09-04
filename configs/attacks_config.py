import os
import yaml
import sys
from pathlib import Path
import torch
import datetime

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
        self.save_path = '/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_name = 'VIT'  #'ptq4vit',RepQ, DeiT, VIT,BEiT,swin_tiny,swin_base Whisper, Owldetection ,Detr, yolos,gpt2,blip,git, other
        model_quant = "Quant_ensemble_static" # Quant, No_Quant

        self.model_name = model_name
        self.second_model_name = "ptq4vit"  # DeiT, VIT, Whisper, Owldetection ,Detr, other

        model_config = {'VIT': 0, 'DeiT': 0,'BEiT_large':1,'BEiT_base':1,  'Whisper': 1, 'Owldetection': 2, 'Detr': 1, 'yolos': 1,"yolos_base":1, 'gpt2': 3,
                        'blip': 3,'swin_tiny': 1,'swin_base':1,
                        "git": 4, 'VIT_large': 0,'VIT_384': 0, 'DeiT_large': 0, "ptq4vit": 0,"RepQ":0,"VIT_small":0,"VIT_large": 0,"VIT_tiny":0}  # 0p

        loss_param_config ={'VIT': [1, 50, 0.01]}

        self.model_config_num = model_config[model_name]

        print("Using model: ", model_name)

        self.loss_func_params = {'MSE': {}}  # BCEWithLogitsLoss , MSE
        self.attack_name = 'PGD'

        self.loss_params = {
            'weights': [loss_param_config["VIT"]]}
        # 0 - loss, 1 - loss on the accuracy, 2 - loss on the total variation [1, 50, 0.01]

        # Mask: implement only for UniversalAttack
        self.mask_option = None  # None,"left_upper64", "left_upper54",'left_upper16', 'left_upper32',  'one_pixel', two_pixel', four_pixel'

        epsilon = 16/255 # 16/255,32/255 12/255
        eps_step = 0.002

        if self.mask_option is not None:
            epsilon = 10000001
            eps_step = 1
            self.image_size, self.image_size = 224, 224


        clip_values = (-1,1) if model_name in ["VIT", "DeiT"] else (-3, 3)
        normalized_std = [0.485, 0.456, 0.406] if model_name in ["ptq4vit","RepQ"] else None
        self.attack_params = {
            'norm': "inf",  # "inf", 2, 1
            'eps': epsilon,
            'eps_step': eps_step,
            'decay': None,
            'max_iter':3000,
            'targeted': True,
            'num_random_init': 1,
            'device': self.device,
            'clip_values': clip_values,
            "normalized_std": normalized_std,  # [0.485, 0.456, 0.406] for imagenets
        }

        ##################################################################################
        # Universal attack
        self.initial_patch = "zeros"  # "random" ,"zero", "ones"
        self.image_size = 224
        self.epochs = 200
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
            'num_workers': 6
        }

        self.model_threshold = 6  # 6
        self.model_threshold_dest = 7 #7
        self.bottom_threshold = 2 #2
        self.target = 70 #70

        self.num_topk_values = 4 # 4
        print("num_topk_values: ", self.num_topk_values)
        self.choice = 0  # 0: topk for each column for each layer, 1: top k from small layers(782)->In reference to all without division , 2: same as 1 but also for large layers(3072)

        self.blocks_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 12 blocks in VIT

        self._set_model(model_name)
        self._set_losses(self.loss_func_params)
        self.mask = self._set_mask()


        with open(ROOT / 'configs/attack_config.yaml', 'r') as stream:
            self.attack_config = yaml.safe_load(stream)[self.attack_name]

        n = f"{model_name}_{model_quant}"
        self.save_path = os.path.join(self.save_path,n)

        self._update_current_dir()

    def __getitem__(self, item):
        return getattr(self, item)

    def _set_model(self, model_name):
        self.model_name = model_name
        with open(ROOT / 'configs/model_config.yaml', 'r') as stream:
            self.model_config = yaml.safe_load(stream)[model_name]

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
        if self.save_path is not None:
            self.current_dir = os.path.join(self.save_path, "experiments", month_name)
        else:
            self.current_dir = os.path.join("experiments", month_name)
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)

    def _set_dataset(self, dataset_name, dataset_path):
        with open(ROOT / dataset_path, 'r') as stream:
            self.dataset_config = yaml.safe_load(stream)[dataset_name]
        self.dataset_config['dataset_name'] = dataset_name

    def _set_mask(self):
        if self.mask_option is None:
            self.patch_size = None
            mask = torch.ones((1, 3, self.image_size, self.image_size), device=self.device)

        elif 'left_upper' in self.mask_option:
            if self.mask_option == 'left_upper64':
                self.patch_size = 64
            if self.mask_option == 'left_upper54':
                self.patch_size = 54
            if self.mask_option == 'left_upper32':
                self.patch_size = 32
            elif self.mask_option == 'left_upper16':
                self.patch_size = 16
            mask_square_location = (0, 0)
            # Define mask
            mask = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)

            mask[:, :, mask_square_location[0]:mask_square_location[0] + self.patch_size,
            mask_square_location[1]:mask_square_location[1] + self.patch_size] = 1
            # 1: in square, 0: out square

        elif self.mask_option == 'one_pixel':
            self.patch_size = 16
            # Create a torch tensor of size (1, 3, 224, 224)
            mask = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)

            # Calculate the center position of each square
            center_positions_x = torch.arange(self.patch_size / 2, self.image_size, self.patch_size)
            center_positions_y = torch.arange(self.patch_size / 2, self.image_size, self.patch_size)

            # Iterate over each square and set the middle pixel to 1
            for x in center_positions_x:
                for y in center_positions_y:
                    mask[:, :, int(x), int(y)] = 1

        elif self.mask_option == 'two_pixels':
            self.patch_size = 16
            mask = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)
            center_positions_x = torch.arange(self.patch_size / 2, self.image_size, self.patch_size)
            center_positions_y = torch.arange(self.patch_size / 2, self.image_size, self.patch_size)
            for x in center_positions_x:
                for y in center_positions_y:
                    mask[:, :, int(x):int(x) + 2, int(y):int(y) + 2] = 1

        elif self.mask_option == 'four_pixels':
            self.patch_size = 16
            mask = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)
            center_positions_x = torch.arange(self.patch_size / 2, self.image_size, self.patch_size)
            center_positions_y = torch.arange(self.patch_size / 2, self.image_size, self.patch_size)
            for x in center_positions_x:
                for y in center_positions_y:
                    mask[:, :, int(x):int(x) + 4, int(y):int(y) + 4] = 1
        else:
            print('mask option not valid')
            mask = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)

        return mask


class OneToOneAttackConfig(BaseConfig):
    def __init__(self):
        super(OneToOneAttackConfig, self).__init__()
        dataset_path = os.path.join(self.root_path, 'demo')

        attack_image_name = 'My_dog2.jpeg'  # egyptian_cat.jpg
        self.attack_img_diff_path = "/dt/shabtaia/dt-fujitsu/datasets/others/imagnet/ILSVRC/Data/DET/amit_use/images/train/n01514668/images/val/n01514668_9964.JPEG"
        target_image_name = 'corgi.jpg'
        self.target_img_orig_path = os.path.join(dataset_path, target_image_name)
        # self.loss_params.update({'images_save_path': os.path.join(self.current_dir, 'outputs')})


class ManyToManyAttackConfig(BaseConfig):
    def __init__(self):
        super(ManyToManyAttackConfig, self).__init__()
        dataset_name = 'imagenet'
        if self.model_name == 'Owldetection':
            dataset_name = 'coco'
        self._set_dataset(dataset_name, 'configs/dataset_config.yaml')


class UniversalAttackConfig(BaseConfig):
    def __init__(self):
        super(UniversalAttackConfig, self).__init__()
        dataset_name = 'imagenet'
        self._set_dataset(dataset_name, 'configs/dataset_config.yaml')


class ClassUniversalAttackConfig(BaseConfig):
    def __init__(self):
        super(ClassUniversalAttackConfig, self).__init__()
        dataset_name = 'imagenet'
        self._set_dataset(dataset_name, 'configs/dataset_class_config.yaml')


config_dict = {
    'Base': BaseConfig,
    'OneToOne': OneToOneAttackConfig,
    'ManyToMany': ManyToManyAttackConfig,
    'Universal': UniversalAttackConfig,
    'ClassUniversal': ClassUniversalAttackConfig
}
