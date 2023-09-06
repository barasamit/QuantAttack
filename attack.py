import time

import pandas as pd
import os
import torch
from utils.model_utils import get_model, get_model_feature_extractor
from utils.init_collect_arrays import outliers_arr, hook_fn,outliers_arr_local

from utils.general import get_instance
from losses import Loss

from torch.profiler import profile, record_function, ProfilerActivity
# from fvcore.nn import flop_count
import warnings
import GPUtil
from utils.attack_utils import count_outliers
import pynvml
warnings.filterwarnings("ignore")


class Attack:
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.attack_type = self.__class__.__name__



        self.model = get_model(cfg, self.cfg['model_name'])
        # saving the relevant layers from here instead of ..../site-packages/transformers/utils/bitsandytes
        for name, module in self.model.named_modules():
            module.register_forward_hook(hook_fn)

        self.feature_extractor = get_model_feature_extractor(self.cfg['model_name'])
        self.model_std = self.feature_extractor.image_std
        self.model_mean = self.feature_extractor.image_mean
        self.clean_time = 0
        self.adv_time = 0
        self.outliers = []

        loss_func = get_instance(self.cfg['losses_config']['module_name'],
                                 self.cfg['losses_config']['class_name'])(**self.cfg['loss_func_params'])

        self.loss = Loss(self.model, [loss_func], "", self.cfg, **self.cfg['loss_params'])
        self.cfg['attack_params']['loss_function'] = self.loss.loss_gradient
        self.cfg['attack_params']['normalized_std'] = self.model_std

        self.attack = get_instance(self.cfg['attack_config']['module_name'],
                                   self.cfg['attack_config']['class_name'])(**self.cfg['attack_params'])


        self.weights = self.cfg.loss_params['weights']
        self.attack_dir = os.path.join(self.cfg['current_dir'], self.get_name())
        self.make_dir(self.attack_dir)
        self.file_name = os.path.join(self.attack_dir, f"results.csv")

    def get_name(self):
        attack_params = self.cfg.attack_params
        k = self.cfg.num_topk_values

        batch_size = self.cfg.loader_params['batch_size']
        b = [str(attack_params['norm']), str(attack_params['eps']), str(attack_params['eps_step']),
             str(attack_params['targeted']), str(attack_params['max_iter']), str(k), str(batch_size),
             str(self.cfg['model_name']),str(self.weights)]

        return '_'.join(b)

    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def calc_model_time(self, x, count_forwards=15):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            rgb = [self.model(x) for _ in range(count_forwards)]
        self.outliers = count_outliers(outliers_arr_local,
                                        threshold=self.cfg.model_threshold)
        end.record()
        torch.cuda.synchronize()
        total_time = (start.elapsed_time(end) / 1000.0) / count_forwards
        outliers_arr_local.clear()

        del rgb  # Free memory used by variable rgb
        torch.cuda.empty_cache()  # Clear up CUDA memory cache
        return total_time

    def calc_acc_topk(self, x_clean, x_adv, k=5):
        p_clean = self.model(x_clean)
        p_adv = self.model(x_adv)

        p_clean_indx = p_clean.logits.argmax(-1)  # model predicts one of the 1000 ImageNet classes
        p_adv_indx = p_adv.logits.argmax(-1)  # model predicts one of the 1000 ImageNet classes
        top1 = torch.equal(p_clean_indx, p_adv_indx)

        try:

            p_adv_topk = torch.topk(p_adv.logits, k, dim=1).indices

            topk = (p_clean_indx.unsqueeze(1) == p_adv_topk).any(dim=1).float().mean().item()

        except:
            topk = 0.0

        return top1, topk

    # def calc_flops(self, x):
    #     flops, _ = flop_count(self.model, x)
    #     return sum(flops.values())

    def get_gpu_temperature(self):
        gpus = GPUtil.getGPUs()
        gpu_temperature = gpus[0].temperature
        return gpu_temperature

    def measure_gpu_power(self, handle, iterr=10):
        before_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)


        after_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

        energy_delta = after_energy - before_energy

        return energy_delta

    def calc_GPU_CPU_time_memory(self, x):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            with record_function("model_inference"):
                try:
                    self.model(x).logits.sum().item()
                except:
                    if self.ids == 0:
                        with torch.no_grad():
                            self.model(x.half())
                    else:
                        with torch.no_grad():
                            self.model(input_ids=self.ids[0], pixel_values=x)
                self.outliers = count_outliers(outliers_arr_local,
                                        threshold=self.cfg.model_threshold)

        # ==== or ====
        cpu_time = prof.profiler.total_average().self_cpu_time_total / 1000.0  # in mil-second
        cuda_time = prof.profiler.total_average().self_cuda_time_total / 1000.0  # in mil-second
        cpu_mem = prof.profiler.total_average().cpu_memory_usage / (2 ** 20)  # in Mbits
        cuda_mem = prof.profiler.total_average().cuda_memory_usage / (2 ** 20)  # in Mbits
        outliers_arr_local.clear()
        return cpu_time, cuda_time, cpu_mem, cuda_mem

    def dict_to_df(self, results):
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.transpose()
        df = pd.concat([df.drop(['clean', 'adv'], axis=1), df['clean'].apply(pd.Series).add_prefix('clean_'),
                        df['adv'].apply(pd.Series).add_prefix('adv_')], axis=1)
        return df

    @torch.no_grad()
    def compute_success(self, x_clean, x_adv, batch_id, img_dir, sava_pd=True, ids=None):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        self.ids = ids
        results = {'batch_id': batch_id, 'clean': dict(), 'adv': dict(), "img_dir": img_dir}
        outliers_arr_local.clear()

        before_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        cpu_time, cuda_time, cpu_mem, cuda_mem = self.calc_GPU_CPU_time_memory(x_adv)
        after_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

        energy_delta = after_energy - before_energy

        results['adv']['power_usage'] = energy_delta

        results['adv']['CUDA_time'] = cuda_time
        results['adv']['CUDA_mem'] = cuda_mem
        results['adv']['CPU_time'] = cpu_time
        results['adv']['CPU_mem'] = cpu_mem
        results['adv']['outliers'] = self.outliers[0]

        # cool_down = 20
        # print(f"strat sleep {cool_down} sec")
        # time.sleep(cool_down)
        # print("end sleep {cool_down} sec")

        # Calculate Clean Memory/Time
        # print("GPU before clean Temperature:", self.get_gpu_temperature(), "°C")
        before_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        cpu_time, cuda_time, cpu_mem, cuda_mem = self.calc_GPU_CPU_time_memory(x_clean)
        after_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        results['clean']['power_usage'] = after_energy - before_energy


        results['clean']['CUDA_time'] = cuda_time
        results['clean']['CUDA_mem'] = cuda_mem
        results['clean']['CPU_time'] = cpu_time
        results['clean']['CPU_mem'] = cpu_mem
        results['clean']['outliers'] = self.outliers[0]

        # Calculate Accuracy & top_k
        try:
            top1, topk = self.calc_acc_topk(x_clean, x_adv)
            results['accuracy'] = int(top1)
            results['topk'] = int(topk)
            print("top1:", top1, "topk:", topk)
        except:
            results['accuracy'] = 0
            results['topk'] = 0


        # print(f"power_usage_clean: {results['clean']['power_usage']}, power_usage_adv: {results['adv']['power_usage']}")
        return self.dict_to_df(results)
