import argparse
import os
import pickle
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from attack import Attack
from configs.attacks_config import config_dict
from utils.data_utils import get_loaders
from utils.general import get_patch
from tqdm import tqdm


# import sys
# sys.path.append('/sise/home/barasa/8_bits_attack/')
# print("torch version: ", torch.__version__)


class UniversalAttack(Attack):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.train_loader, self.val_loader, self.test_loader = get_loaders(self.cfg.loader_params,
                                                                           self.cfg.dataset_config,
                                                                           ['validation', 'test'],
                                                                           model_name=self.cfg.model_name)
        self.train_loader = self.val_loader

        self.number_of_train_images = cfg.number_of_training_images
        self.number_of_val_images = cfg.number_of_val_images
        self.number_of_val_images = cfg.number_of_val_images
        self.number_of_test_images = cfg.number_of_test_images
        self.csv_name = "universal_attack.csv"
        self.save_changes = True
        self.weights = self.cfg.loss_params['weights']
        self.model_name = self.cfg.model_name
        self.lr = self.attack.eps_step
        self.cls = None
        self.start_num = 0

    def generate(self):
        if self.cls is None:
            main_folder = os.path.join("/sise/home/barasa/8_bits_attack/experiments/Final_results/Universal/",
                                       self.model_name)
            self.name = f"lr_{self.lr.item()}_epsilon_{self.attack.eps.item()}_norm_{self.attack.norm}_weights{self.weights}_name{self.model_name}.csv"
        else:
            main_folder = os.path.join("/sise/home/barasa/8_bits_attack/experiments/Final_results/Class_universal/",
                                       self.model_name)
            self.name = f"lr_{self.lr.item()}_epsilon_{self.attack.eps.item()}_norm_{self.attack.norm}_weights{self.weights}_cls_{self.cls}_name{self.model_name}.csv"
        print("Starting Universal attack...")
        print("###############################################")
        print("parameters:")
        print("lr: ", self.lr)
        print("epsilon: ", self.attack.eps)
        print("number of training images: ", self.number_of_train_images)
        print("number of validation images: ", self.number_of_val_images)
        print("number of test images: ", self.number_of_test_images)

        print("###############################################")

        # create dir
        # Path(os.path.join(self.cfg.current_dir, self.name)).mkdir(parents=True, exist_ok=True)
        # main_dir = os.path.join(self.cfg.current_dir, self.name)
        # Path(os.path.join(main_dir, "images")).mkdir(parents=True, exist_ok=True)

        results_combine = pd.DataFrame()


        files = os.listdir(main_folder)[self.start_num:]
        for file in files:
            main_dir = os.path.join(main_folder, file)
            try:
                adv_pert_cpu = torch.load(os.path.join(main_dir, "perturbation_torch.pt"))
            except:
                continue
            if self.cls:
                if file.split("_")[-2] != self.cls:
                    continue

            adv_pert_cpu = torch.load("/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/March/Universal_None_lr_0.002_epsilon_0.125490203499794_norm_inf_weights[[1, 50.0, 0.0]]_nameDeiT_images_250.csv/perturbation_torch.pt")

            dir = os.path.join(main_dir, "new_" + self.csv_name)
            num_rows = None
            # to skip already computed images
            if os.path.exists(dir):
                try:
                    results_combine = pd.read_csv(dir)
                    results_combine = results_combine.drop(results_combine.index[-1])
                    num_rows = len(results_combine)
                except:
                    pass
            with torch.no_grad():
                for batch_id, (batch, img_dir, _) in enumerate(tqdm(self.test_loader)):
                    if num_rows and batch_id < num_rows:
                        continue
                    print(f"batch{batch_id}")
                    if batch_id >= self.number_of_test_images: break  # self.cfg.test_success_size
                    batch = batch.to(self.attack.device)
                    adv_pert_gpu = adv_pert_cpu.to(self.attack.device)
                    images_with_pert = self.apply_perturbation(batch, adv_pert_gpu)
                    if len(images_with_pert.shape) > 4:
                        images_with_pert = images_with_pert.squeeze(0)
                    if len(batch.shape) > 4:
                        batch = batch.squeeze(0)
                    results = self.compute_success(batch, images_with_pert, batch_id, img_dir)
                    results_combine = pd.concat([results_combine, results], axis=0)
                    print("saving results... at ", os.path.join(main_dir, self.csv_name))
                    results_combine.to_csv(os.path.join(main_dir, "new_" + self.csv_name), index=False)

                    if batch_id % 10 == 0:
                        img1 = batch
                        img2 = images_with_pert

                        # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
                        save_image(self.denormalize(img1),
                                   os.path.join(main_dir, "images", "{}_clean.png".format(batch_id)))
                        save_image(self.denormalize(img2),
                                   os.path.join(main_dir, "images", "{}_adv.png".format(batch_id)))

            torch.save(adv_pert_cpu, os.path.join(main_dir, "perturbation_torch.pt"))
            save_image(torch.clamp(adv_pert_cpu, 0, 1), os.path.join(main_dir, "images", "perturbation.png"))

            results_combine.to_csv(os.path.join(main_dir, "new_" + self.csv_name), index=False)
            print("saved ", self.csv_name)

    def forward_step(self, adv_pert_cpu, images, targets):
        adv_pert_device = adv_pert_cpu.to(self.cfg['device'])
        images = images.to(self.cfg['device'])
        adv_batch = self.apply_perturbation(images, adv_pert_device)

        loss, outliers = self.loss.loss_gradient(adv_batch, images, 0, "universal")

        return loss, loss.item(), outliers, adv_batch

    def denormalize(self, x, mean=None, std=None):
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

    def apply_perturbation(self, images, adv_pert):

        adv_batch = images + adv_pert  # no need to multiply by step because it happens in the optimizer.step

        adv_batch.data.clamp_(-1, 1)
        return adv_batch

    def _projection(self, values):
        tol = 10e-8

        values_tmp = values.reshape(values.shape[0], -1)
        if self.attack.norm == 2:
            values_tmp = (values_tmp *
                          torch.min(
                              torch.tensor([1.0], dtype=torch.float32).to(values.device),
                              self.attack.eps.to(values.device) / (torch.norm(values_tmp, p=2, dim=1) + tol),
                          ).unsqueeze_(-1)
                          )
        elif self.attack.norm == 1:
            values_tmp = (values_tmp *
                          torch.min(
                              torch.tensor([1.0], dtype=torch.float32).to(values.device),
                              self.attack.eps.to(values.device) / (torch.norm(values_tmp, p=1, dim=1) + tol),
                          ).unsqueeze_(-1)
                          )
        elif self.attack.norm == 'inf':
            values_tmp = values_tmp.sign() * torch.min(values_tmp.abs(), self.attack.eps.to(values.device)).to(
                values.device)
        values = values_tmp.reshape(values.shape)
        return values

    def _compute_perturbation(self, adv_x, targets, momentum):
        grad = adv_x.grad.to(self.attack.device)
        tol = 10e-8
        # Apply norm
        if self.attack.norm == "inf":
            grad = grad.sign()
        elif self.attack.norm == 1:
            ind = tuple(range(1, len(adv_x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdim=True) + tol)
        elif self.attack.norm == 2:
            ind = tuple(range(1, len(adv_x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, dim=ind, keepdim=True)) + tol)

        if self.attack.normalized_std:
            grad.index_copy_(1, torch.LongTensor([0]).to(self.attack.device),
                             grad.index_select(1, torch.LongTensor([0]).to(self.attack.device)) /
                             self.attack.normalized_std[0])
            grad.index_copy_(1, torch.LongTensor([1]).to(self.attack.device),
                             grad.index_select(1, torch.LongTensor([1]).to(self.attack.device)) /
                             self.attack.normalized_std[1])
            grad.index_copy_(1, torch.LongTensor([2]).to(self.attack.device),
                             grad.index_select(1, torch.LongTensor([2]).to(self.attack.device)) /
                             self.attack.normalized_std[2])

        return grad

    @torch.no_grad()
    def evaluate(self, adv_pert):
        running_loss = []
        number_of_images = self.number_of_val_images
        progress_bar = tqdm(enumerate(self.val_loader), desc=f'Eval',
                            total=min(len(self.val_loader), number_of_images), ncols=150)
        prog_bar_desc = 'Batch Loss: {:.6}'
        for i_batch, (images, labels, _) in progress_bar:
            if i_batch >= number_of_images: break  # was eval_size
            # with torch.no_grad():
            #     predict = self.model(images.to(self.attack.device))
            if len(images.shape) > 4:
                images = images.squeeze(0)

            loss, cur_loss, outliers = self.forward_step(adv_pert, images, labels)
            running_loss.append(cur_loss)
            progress_bar.set_postfix_str(
                "Batch Loss: {:.6} | Outliers: {}".format(sum(running_loss) / (i_batch + 1), outliers))
            # progress_bar.set_postfix_str(prog_bar_desc.format(sum(running_loss) / (i_batch + 1)))

        return running_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Many-to-Many Attack')
    parser.add_argument('--accuracy_loss', type=float, default=0, help='Weight for accuracy loss')
    parser.add_argument('--TV_loss', type=float, default=0, help='Weight for Total Variation loss')
    parser.add_argument('--cls', type=str, default='n01495701',
                        help='class to attack')  # ["n01495701", "n01531178", "n01644900", "n01688243", "n06874185"]
    parser.add_argument('--file_num', type=int, default=0, help='from where to start')
    print(parser.parse_args())
    return parser.parse_args()


def main():
    args = parse_args()

    config_type = 'Universal'
    cfg = config_dict[config_type]()
    accuracy_loss = args.accuracy_loss
    TV_loss = args.TV_loss

    cfg.loss_params = {'weights': [[1, accuracy_loss,
                                    TV_loss]]}  # 0 - loss, 1 - loss on the accuracy, 2 - loss on the total variation [1, 50, 0.01]

    attack = UniversalAttack(cfg)
    attack.start_num = args.file_num
    attack.generate()


def Class_main():
    args = parse_args()
    root_path = "/dt/shabtaia/dt-fujitsu/datasets/others/imagnet/ILSVRC/Data/DET/amit_use/images/train/n01495701"
    config_type = 'ClassUniversal'
    with open("./configs/class_temp.yaml", 'r') as stream:
        yaml_data = yaml.safe_load(stream)
        updated_root_path = os.path.join(root_path.rsplit("/", 1)[0], args.cls)
        yaml_data['imagenet']['root_path'] = updated_root_path
        os.remove("/sise/home/barasa/8_bits_attack/configs/dataset_class_config.yaml")
        with open("./configs/dataset_class_config.yaml", 'w') as outfile:
            yaml.dump(yaml_data, outfile, default_flow_style=False)

    cfg = config_dict[config_type]()
    accuracy_loss = args.accuracy_loss
    TV_loss = args.TV_loss
    cfg.loss_params = {'weights': [[1, accuracy_loss,
                                    TV_loss]]}  # 0 - loss, 1 - loss on the accuracy, 2 - loss on the total variation [1, 50, 0.01]

    attack = UniversalAttack(cfg)
    attack.cls = args.cls
    attack.generate()


if __name__ == '__main__':
    ##############################
    main()
    # Class_main()
