import pandas as pd
import os
import torch
import cv2
# from utils.model_utils import get_lensless_model, get_classification_model
from utils.model_utils import get_vit_model, get_vit_feature_extractor
from main_ViT import outliers_arr, hook_fn

from utils.general import get_instance, save_class_to_file, crop_images
from losses import Loss
from utils.general import preplot, process_imgs, auroc_aupr_scores
import numpy as np
import time
from statistics import mean

from torch.profiler import profile, record_function, ProfilerActivity
from fvcore.nn import flop_count
import warnings

warnings.filterwarnings("ignore")


class Attack:
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.attack_type = self.__class__.__name__

        self.attack_dir = os.path.join(self.cfg['current_dir'], self.get_name())
        self.make_dir(self.attack_dir)
        self.file_name = os.path.join(self.attack_dir, "results.csv")

        self.model = get_vit_model(cfg)
        # saving the relevant layers from here instead of ..../site-packages/transformers/utils/bitsandytes
        for name, module in self.model.named_modules():
            module.register_forward_hook(hook_fn)

        self.feature_extractor = get_vit_feature_extractor()
        self.clean_time = 0
        self.adv_time = 0
        self.outliers = []

        s = ''
        print('Using the following distance functions in the loss:' + s)
        loss_func = get_instance(self.cfg['losses_config']['module_name'],
                                 self.cfg['losses_config']['class_name'])(**self.cfg['loss_func_params'])

        self.loss = Loss(self.model, [loss_func], "",self.cfg.attack_type,self.cfg.attack_params["max_iter"], **self.cfg['loss_params'])
        self.cfg['attack_params']['loss_function'] = self.loss.loss_gradient
        self.attack = get_instance(self.cfg['attack_config']['module_name'],
                                   self.cfg['attack_config']['class_name'])(**self.cfg['attack_params'])

        # save_class_to_file(self.cfg, self.cfg['current_dir'])

        self.gts = {i: [] for i in range(0, self.cfg['estimator_config']['num_of_classes'])}
        self.preds_clean = {i: [] for i in range(0, self.cfg['estimator_config']['num_of_classes'])}
        self.preds_adv = {i: [] for i in range(0, self.cfg['estimator_config']['num_of_classes'])}

    def get_name(self):
        a = self.cfg.attack_params
        b = [str(a['norm']), str(a['eps']), str(a['eps_step']), str(a['targeted'])]
        return '_'.join(b)

    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)



    def register_loss_values(self, batch_id):
        batch_result = pd.Series([f'batch{batch_id}'] + self.attack.loss_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'), mode='a',
                                         header=False, index=False)

    def calc_model_time(self, x, count_forwards=15):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        rgb = [self.model(x) for _ in range(count_forwards)]
        self.outliers = outliers_arr
        end.record()
        torch.cuda.synchronize()
        total_time = (start.elapsed_time(end) / 1000.0) / count_forwards
        outliers_arr.clear()
        return total_time

    def calc_acc_topk(self, x_clean, x_adv, k=5):
        p_clean = self.model(x_clean)
        p_adv = self.model(x_adv)

        p_clean_indx = p_clean.logits.argmax(-1)  # model predicts one of the 1000 ImageNet classes
        p_adv_indx = p_adv.logits.argmax(-1)  # model predicts one of the 1000 ImageNet classes

        p_adv_topk = torch.topk(p_adv.logits, k, dim=1).indices

        top1 = torch.equal(p_clean_indx, p_adv_indx)
        topk = (p_clean_indx.unsqueeze(1) == p_adv_topk).any(dim=1).float().mean().item()

        return top1, topk

    def calc_flops(self, x):
        flops, _ = flop_count(self.model, x)
        return sum(flops.values())

    def calc_GPU_CPU_time_memory(self, x):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            with record_function("model_inference"):
                self.model(x)
                self.outliers = outliers_arr.copy()

        # ==== or ====
        cpu_time = prof.profiler.total_average().self_cpu_time_total / 1000.0  # in mil-second
        cuda_time = prof.profiler.total_average().self_cuda_time_total / 1000.0  # in mil-second
        cpu_mem = prof.profiler.total_average().cpu_memory_usage / (2 ** 20)  # in Mbits
        cuda_mem = prof.profiler.total_average().cuda_memory_usage / (2 ** 20)  # in Mbits
        outliers_arr.clear()
        return cpu_time, cuda_time, cpu_mem, cuda_mem

    def dict_to_df(self, results):
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.transpose()
        df = pd.concat([df.drop(['clean', 'adv'], axis=1), df['clean'].apply(pd.Series).add_prefix('clean_'),
                        df['adv'].apply(pd.Series).add_prefix('adv_')], axis=1)
        return df

    @torch.no_grad()
    def compute_success(self, x_clean, x_adv, batch_id, img_dir, sava_pd=True):
        results = {'batch_id': batch_id, 'clean': dict(), 'adv': dict(), "img_dir": img_dir}

        # Calculate Adv Memory/Time
        cpu_time, cuda_time, cpu_mem, cuda_mem = self.calc_GPU_CPU_time_memory(x_adv)
        results['adv']['CUDA_time'] = cuda_time
        results['adv']['CUDA_mem'] = cuda_mem
        results['adv']['CPU_time'] = cpu_time
        results['adv']['CPU_mem'] = cpu_mem
        results['adv']['outliers'] = sum([len(o) for o in self.outliers])

        # Calculate Clean Memory/Time
        cpu_time, cuda_time, cpu_mem, cuda_mem = self.calc_GPU_CPU_time_memory(x_clean)
        results['clean']['CUDA_time'] = cuda_time
        results['clean']['CUDA_mem'] = cuda_mem
        results['clean']['CPU_time'] = cpu_time
        results['clean']['CPU_mem'] = cpu_mem
        results['clean']['outliers'] = sum([len(o) for o in self.outliers])

        # Calculate Accuracy & top_k
        top1, topk = self.calc_acc_topk(x_clean, x_adv)
        results['accuracy'] = int(top1)
        results['topk'] = int(topk)

        # # Calculate GPLOPs
        # results['adv']['GFLOPs'] = self.calc_flops(x_adv)
        # results['clean']['GFLOPs'] = self.calc_flops(x_clean)

        return self.dict_to_df(results)

    def calculate_final_metrics(self):
        metrics_df = pd.read_csv(os.path.join(self.cfg['current_dir'], 'acc_results.csv'),
                                 header=0,
                                 index_col=0)
        clean_acc_mean = metrics_df['clean_acc'].mean()
        ut_acc_mean = metrics_df['ut_adv_acc'].mean()
        t_acc_mean = metrics_df['t_adv_acc'].mean()
        auroc_clean = auroc_aupr_scores(np.stack(list(self.gts.values()), axis=1),
                                        np.stack(list(self.preds_clean.values()), axis=1),
                                        average_types=['macro'])['macro']
        auroc_adv = auroc_aupr_scores(np.stack(list(self.gts.values()), axis=1),
                                      np.stack(list(self.preds_adv.values()), axis=1),
                                      average_types=['macro'])['macro']

        with open(os.path.join(self.cfg['current_dir'], 'final_results.txt'), 'w') as f:
            f.write('Clean Accuracy,Source Label Accuracy,Target Label Accuracy,Clean AuROC,Adv Auroc\n')
            f.write(f'{clean_acc_mean},{ut_acc_mean},{t_acc_mean},{auroc_clean},{auroc_adv}')

        return clean_acc_mean, ut_acc_mean, t_acc_mean, auroc_clean, auroc_adv

    def print_metrics(self, clean_acc_mean, ut_acc_mean, t_acc_mean, auroc_clean, auroc_adv):
        print('Average accuracy for clean images with source labels: {}%'.format(
            str(round(clean_acc_mean * 100, 2))))
        print('Average accuracy for adv images with source labels: {}%'.format(
            str(round(ut_acc_mean * 100, 2))))
        print('Average accuracy for adv images with target labels: {}%'.format(
            str(round(t_acc_mean * 100, 2))))
        print('AuROC for clean images with source labels: {}%'.format(
            str(round(auroc_clean * 100, 2))))
        print('AuROC for adv images with source labels: {}%'.format(
            str(round(auroc_adv * 100, 2))))
