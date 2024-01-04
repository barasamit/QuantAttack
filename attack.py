import time

import pandas as pd
import os
import torch
from transformers import YolosFeatureExtractor

from utils.model_utils import get_model, get_model_feature_extractor
from utils.init_collect_arrays import outliers_arr, hook_fn, outliers_arr_local

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
        self.second_model = None
        self.model = get_model(cfg, self.cfg['model_name'])
        if self.cfg['second_model_name'] is not None:
            self.second_model = get_model(cfg, self.cfg['second_model_name'])
            for name, module in self.second_model.named_modules():
                module.register_forward_hook(hook_fn)
        # saving the relevant layers from here instead of ..../site-packages/transformers/utils/bitsandytes


        for name, module in self.model.named_modules():
            module.register_forward_hook(hook_fn)

        # for param in self.model.parameters():
        #     try:
        #         param.requires_grad = True
        #     except:
        #         print(param)
        #
        # for param in self.model.base_model.parameters():
        #     try:
        #         param.requires_grad = True
        #     except:
        #         pass

        self.feature_extractor = get_model_feature_extractor(self.cfg['model_name'])
        self.model_std = [0.5, 0.5, 0.5]
        self.model_mean = [0.5, 0.5, 0.5]
        if self.cfg['model_name'] in ['VIT', "DeiT", 'Detr']:
            self.model_std = self.feature_extractor.image_std
            self.model_mean = self.feature_extractor.image_mean
        self.clean_time = 0
        self.adv_time = 0
        self.outliers = []

        loss_func = get_instance(self.cfg['losses_config']['module_name'],
                                 self.cfg['losses_config']['class_name'])(**self.cfg['loss_func_params'])

        self.loss = Loss(self.model, [loss_func], "", self.cfg, self.second_model, **self.cfg['loss_params'])
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
        target = self.cfg.target

        batch_size = self.cfg.loader_params['batch_size']
        b = [str(attack_params['norm']), str(attack_params['eps']), str(attack_params['eps_step']),
             str(attack_params['targeted']), str(attack_params['max_iter']), str(k), str(batch_size),
             str(self.cfg['model_name']), str(self.weights), str(target)]

        return '_'.join(b)

    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def calc_acc_topk(self, x_clean, x_adv, k=5):
        try:
            p_clean = self.model(x_clean)
            p_adv = self.model(x_adv)
        except:
            feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
            p_clean = self.model(x_clean.half())
            p_adv = self.model(x_adv.half())
            results_c = feature_extractor.post_process_object_detection(p_clean, threshold=0.9)[0]
            results_adv = feature_extractor.post_process_object_detection(p_adv, threshold=0.9)[0]

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
                outliers_arr_local.clear()
                try:
                    self.model(x).logits.sum().item()
                    # self.model(input_ids = torch.randint(0, 100, (10,10)),pixel_values=x)
                except:
                    # self.model(x).logits.sum().item()
                    if self.ids is None or self.ids == [] or self.ids == torch.tensor([0]):

                        with torch.no_grad():
                            self.model(x.half())
                    else:
                        with torch.no_grad():
                            try:
                                self.model(input_ids=self.ids[0], pixel_values=x)
                            except:
                                self.model(input_ids = torch.randint(0, 100, (10,10)),pixel_values=x)

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
        results = {'batch_id': batch_id, 'clean': dict(), 'adv': dict(), "img_dir": img_dir}
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        self.ids = ids
        def gpu_warm_up():
            for _ in range(2):
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                before_energy_clean = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                # empty tensor size (1,3,224,224)to warm up GPU
                # tensor = torch.zeros((1, 3, 224, 224)).cuda()
                _ = self.calc_GPU_CPU_time_memory(x_clean)
                after_energy_clean = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            torch.cuda.synchronize()
        gpu_warm_up()
        outliers_arr_local.clear()

        # Initialize variables to accumulate measurements
        total_energy_clean = 0
        total_energy_adv = 0
        total_CUDA_time_clean = 0
        total_CUDA_time_adv = 0
        total_CUDA_mem_clean = 0
        total_CUDA_mem_adv = 0
        total_CPU_time_clean = 0
        total_CPU_time_adv = 0
        total_CPU_mem_clean = 0
        total_CPU_mem_adv = 0
        total_outliers_clean = 0
        total_outliers_adv = 0
        total_accuracy = 0
        total_topk = 0

        num_measurements = 1  # Number of times to repeat the calculation

        for _ in range(num_measurements):

            # Measure adversarial example
            try:

                before_energy_adv = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                cpu_time_adv, cuda_time_adv, cpu_mem_adv, cuda_mem_adv = self.calc_GPU_CPU_time_memory(x_adv)
                after_energy_adv = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                total_outliers_adv += self.outliers[0]

                # Measure clean example
                before_energy_clean = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                cpu_time_clean, cuda_time_clean, cpu_mem_clean, cuda_mem_clean = self.calc_GPU_CPU_time_memory(x_clean)
                after_energy_clean = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                total_outliers_clean += self.outliers[0]

                # Accumulate measurements for clean and adv separately
                total_energy_clean += (after_energy_clean - before_energy_clean)
                total_energy_adv += (after_energy_adv - before_energy_adv)
                total_CUDA_time_clean += cuda_time_clean
                total_CUDA_time_adv += cuda_time_adv
                total_CUDA_mem_clean += cuda_mem_clean
                total_CUDA_mem_adv += cuda_mem_adv
                total_CPU_time_clean += cpu_time_clean
                total_CPU_time_adv += cpu_time_adv
                total_CPU_mem_clean += cpu_mem_clean
                total_CPU_mem_adv += cpu_mem_adv

                try:
                    top1, topk = self.calc_acc_topk(x_clean, x_adv)
                    total_accuracy += top1
                    total_topk += topk
                except:
                    pass
            except Exception as e:
                print(e)
                continue
        # Calculate averages for clean and adv
        results['clean']['power_usage'] = total_energy_clean / num_measurements
        results['clean']['CUDA_time'] = total_CUDA_time_clean / num_measurements
        results['clean']['CUDA_mem'] = total_CUDA_mem_clean / num_measurements
        results['clean']['CPU_time'] = total_CPU_time_clean / num_measurements
        results['clean']['CPU_mem'] = total_CPU_mem_clean / num_measurements
        results['clean']['outliers'] = total_outliers_clean / num_measurements

        results['adv']['power_usage'] = total_energy_adv / num_measurements
        results['adv']['CUDA_time'] = total_CUDA_time_adv / num_measurements
        results['adv']['CUDA_mem'] = total_CUDA_mem_adv / num_measurements
        results['adv']['CPU_time'] = total_CPU_time_adv / num_measurements
        results['adv']['CPU_mem'] = total_CPU_mem_adv / num_measurements
        results['adv']['outliers'] = total_outliers_adv / num_measurements

        # Calculate average accuracy and topk
        results['accuracy'] = int(total_accuracy / num_measurements)
        results['topk'] = int(total_topk / num_measurements)

        return self.dict_to_df(results)

    def compute_success2(self, x_clean, x_adv_list, batch_id, img_dir, sava_pd=True, ids=None):

        results = {'batch_id': batch_id, 'clean': dict(), 'adv': [], "img_dir": img_dir}
        self.ids = ids
        # Function to calculate the average of a list of values
        def average_list(lst):
            if len(lst) == 0:
                return 0
            return sum(lst) / len(lst)

        # Function for GPU warm-up

        def gpu_warm_up():
            for _ in range(3):
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                before_energy_clean = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                _ = self.calc_GPU_CPU_time_memory(x_clean)
                after_energy_clean = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            torch.cuda.synchronize()
        gpu_warm_up()

        outliers_arr_local.clear()

        # Initialize results dictionaries
        clean_results = {
            'power_usage': [],
            'CUDA_time': [],
            'CUDA_mem': [],
            'outliers': []
        }
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        torch.cuda.empty_cache()
        before_energy_clean = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        cpu_time_clean, cuda_time_clean, cpu_mem_clean, cuda_mem_clean = self.calc_GPU_CPU_time_memory(x_clean)
        after_energy_clean = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        total_outliers_clean = self.outliers[0]
        clean_results['power_usage'].append(after_energy_clean - before_energy_clean)
        clean_results['CUDA_time'].append(cuda_time_clean)
        clean_results['CUDA_mem'].append(cuda_mem_clean)
        clean_results['outliers'].append(total_outliers_clean)
        torch.cuda.empty_cache()
        outliers_arr_local.clear()
        adv_results_list = [{} for _ in x_adv_list]

        for _ in range(1):
            # Process clean example


            # Process adversarial examples
            for i, x_adv in enumerate(x_adv_list):
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                torch.cuda.empty_cache()
                before_energy_adv = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                cpu_time_adv, cuda_time_adv, cpu_mem_adv, cuda_mem_adv = self.calc_GPU_CPU_time_memory(x_adv)
                after_energy_adv = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                total_outliers_adv = self.outliers[0]
                torch.cuda.empty_cache()

                if _ == 0:
                    adv_results_list[i] = {
                        'power_usage': [],
                        'CUDA_time': [],
                        'CUDA_mem': [],
                        'outliers': [],
                        'accuracy': []
                    }

                adv_results_list[i]['power_usage'].append(after_energy_adv - before_energy_adv)
                adv_results_list[i]['CUDA_time'].append(cuda_time_adv)
                adv_results_list[i]['CUDA_mem'].append(cuda_mem_adv)
                adv_results_list[i]['outliers'].append(total_outliers_adv)

                try:
                    top1, topk = self.calc_acc_topk(x_clean, x_adv)
                    adv_results_list[i]['accuracy'].append(top1)
                except Exception as e:
                    print(e)
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # torch.cuda.empty_cache()
        # before_energy_clean = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        # cpu_time_clean, cuda_time_clean, cpu_mem_clean, cuda_mem_clean = self.calc_GPU_CPU_time_memory(x_clean)
        # after_energy_clean = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        # total_outliers_clean = self.outliers[0]
        # clean_results['power_usage'].append(after_energy_clean - before_energy_clean)
        # clean_results['CUDA_time'].append(cuda_time_clean)
        # clean_results['CUDA_mem'].append(cuda_mem_clean)
        # clean_results['outliers'].append(total_outliers_clean)
        # torch.cuda.empty_cache()
        # outliers_arr_local.clear()

        # Calculate the average of measurements for the clean example
        avg_clean_result = {
            'power_usage': average_list(clean_results['power_usage']),
            'CUDA_time': average_list(clean_results['CUDA_time']),
            'CUDA_mem': average_list(clean_results['CUDA_mem']),
            'outliers': average_list(clean_results['outliers'])
        }

        results['clean'] = avg_clean_result

        # Calculate the average of measurements for the adversarial examples
        for adv_results in adv_results_list:
            avg_adv_result = {
                'power_usage': average_list(adv_results['power_usage']),
                'CUDA_time': average_list(adv_results['CUDA_time']),
                'CUDA_mem': average_list(adv_results['CUDA_mem']),
                'outliers': average_list(adv_results['outliers']),
                'accuracy': average_list(adv_results['accuracy'])
            }
            results['adv'].append(avg_adv_result)

        torch.cuda.empty_cache()

        return self.dict_to_df(results)

