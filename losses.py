from pathlib import Path

import torch
from utils.attack_utils import count_outliers
from utils.general import save_graph, print_outliers
from utils.init_collect_arrays import input_arr, outliers_arr, outliers_arr_local, all_act, layer_norm_arr, layer_norm_module
from utils.losses_utils import clear_lists, stack_tensors_with_same_shape
import torch.nn as nn
# from torchvision.utils import save_image
from torchmetrics.image import TotalVariation
import pandas as pd
import random

random.seed(42)
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)


class Loss:

    def __init__(self, model, loss_fns, convert_fn, cfg, second_model, images_save_path=None, mask=None,
                 weights=None, iteration=0,
                 **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.second_model = second_model
        self.second = False
        if self.second_model is not None:
            self.second = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fns = loss_fns
        self.convert_fn = convert_fn
        self.images_save_path = images_save_path
        self.mask = mask
        self.tv = TotalVariation().to(self.device)
        self.iteration = iteration
        self.max_iter = cfg.attack_params["max_iter"]
        self.attack_type = cfg.attack_type
        if self.images_save_path is not None:
            Path(self.images_save_path).mkdir(parents=True, exist_ok=True)
        if weights is not None:
            self.loss_weights = weights
        else:
            self.loss_weights = [1] * len(loss_fns)
        self.means_array = []
        self.weigths = []
        self.biass = []
        self.total_outliers_list = []
    def get_input_targeted(self, matmul_lists):
        # Get the batch size, rows and columns -> if use other vit model, change the shape and maybe more shapes
        stacked_tensors = stack_tensors_with_same_shape(matmul_lists)

        # Get the top k values
        threshold = self.cfg.model_threshold_dest
        bottom_threshold = self.cfg.bottom_threshold

        selected_values_list = []
        targets_list = []

        for i, tensor in enumerate(stacked_tensors):
            # attack specific layers
            if len(tensor.shape) < 3: continue
            tensor = tensor.abs()
            try:
                t_max = tensor.topk(self.cfg.num_topk_values, dim=2)[0]
            except:
                continue

            mask_lower = t_max > bottom_threshold
            mask_upper = t_max < threshold
            # Combine the two masks using the logical AND operator &
            mask = mask_lower & mask_upper
            # mask = mask_upper

            selected_values = t_max[mask]
            if len(selected_values) == 0:
                selected_values = torch.tensor([self.cfg.target]).to(self.device)
            target = torch.full_like(selected_values, self.cfg.target)

            selected_values_list.append(selected_values)
            targets_list.append(target)
            del tensor
        torch.cuda.empty_cache()

        return selected_values_list, targets_list

    def denormalize(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

    def loss_gradient(self, x, y, ids=None, loss_type="many_to_many"):
        input_arr.clear()
        x_grad = self.prepare_x_grad(x, loss_type)

        # for combine models

        pred, second = self.predict(x_grad, ids,self.second)
        matmul_lists = input_arr.copy()

        self.iteration += 1
        # for each block
        # b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12 = matmul_lists[0:6],matmul_lists[6:12],matmul_lists[12:18],matmul_lists[18:24],matmul_lists[24:30],matmul_lists[30:36],matmul_lists[36:42],matmul_lists[42:48],matmul_lists[48:54],matmul_lists[54:60],matmul_lists[60:66],matmul_lists[66:72]
        # inputs, targets = self.get_input_targeted(b1+b2+b3+b4+b5+b6+b7+b8+b9)
        inputs, targets = self.get_input_targeted(matmul_lists)

        total_outliers, outs_ratio = count_outliers(outliers_arr_local, threshold=self.cfg.model_threshold)

        if self.should_print_outliers():
            try:
                outliers_df = print_outliers(matmul_lists, outs_ratio, self.get_number_of_blocks())
                print()
                print(outliers_df)
            except:
                pass


        loss = torch.zeros(1, device="cuda")
        loss = self.calculate_loss(inputs, targets, x_grad, y, ids, second, pred)
        clear_lists(input_arr, outliers_arr, outliers_arr_local, all_act, matmul_lists, layer_norm_arr, layer_norm_module)

        if loss_type == "universal":
            return loss, total_outliers
        # loss to full precision
        grads, loss_val = self.backward_and_cleanup(loss, x_grad)
        return grads, loss_val, total_outliers

    def prepare_x_grad(self, x, loss_type):
        if loss_type == "universal":
            x_grad = x.clone().requires_grad_(True)
        else:
            x_grad = x.clone().detach().requires_grad_(True)
        return x_grad

    def predict(self, x_grad, ids, second=False):
        second = second
        if self.cfg.model_config_num == 0:
            if second:
                if random.random() < 0.5:
                    pred = self.model(x_grad)
                else:
                    pred = self.second_model(x_grad)
            else:
                pred = self.model(x_grad)
        elif self.cfg.model_config_num == 1:
            pred = self.model(x_grad.half())  # for wisper model
        elif self.cfg.model_config_num == 2:
            pred = self.model(input_ids=ids[0], pixel_values=x_grad)  # for Owldetection model
        elif self.cfg.model_config_num == 3:  # for image captioning model
            gen_kwargs = {"max_length": 16, "num_beams": 4}
            pred = self.model.generate(x_grad, **gen_kwargs)
        elif self.cfg.model_config_num == 4:  # for image captioning model
            pred = self.model(input_ids=torch.randint(0, 100, (10, 10)), pixel_values=x_grad)
        else:
            pred = self.model(x_grad)
        if second:
            pred_2 = self.second_model(x_grad)
            # randomly select the model to use
            if random.random() < 0.5:
                return pred, second
            else:
                return pred_2, pred
        return pred, None

    def should_print_outliers(self):
        return self.iteration % 200 == 0 or self.iteration == 0

    def get_number_of_blocks(self):
        if hasattr(self.model.config, 'num_attention_heads'):
            return self.model.config.num_hidden_layers
        else:
            return self.model.config.text_config.num_hidden_layers

    def calculate_loss(self, inputs, targets, x_grad, y, ids, second, pred):
        loss = torch.zeros(1, device="cuda")
        for loss_fn, loss_weight in zip(self.loss_fns, self.loss_weights):
            loss = self.add_various_losses(loss, loss_fn, loss_weight, inputs, targets, x_grad, y, ids, second, pred)
        return loss

    def add_various_losses(self, loss, loss_fn, loss_weight, inputs, targets, x_grad, y, ids, second, pred):
        for i in range(len(inputs)):
            # temp_loss = loss_fn(inputs[i], targets[i]).squeeze().mean()
            # loss.add_(loss_weight[0] * temp_loss)
            # del temp_loss
            try:
                temp_loss = -inputs[i].mean()
                loss.add_(loss_weight[0] * temp_loss)
                del temp_loss
            except:
                pass


        #
        # # add loss for accuracy
        if loss_weight[1] != 0:
            true_label, y_tag = self.get_true_label_and_y_tag(y, ids, second, pred)
            l = loss_fn(y_tag.squeeze().float(), true_label.squeeze().float()).mean()
            loss.add_(loss_weight[1] * l)

        #
        # # add loss for total variation
        # if loss_weight[2] != 0:
        #     loss += loss_weight[2] * self.tv(x_grad)

        # add loss for variance
        # variances = [t.var(dim=-1, keepdim=True,unbiased=False).squeeze().var() for t in layer_norm_arr[:]]
        # variances = [i for i in variances if not torch.isinf(i) and i.item() > 0]
        # var_mean = torch.stack(variances).mean()
        # loss += var_mean
        #
        # # Add loss for the mean
        # if loss_weight[2] != 0:
        #     means = [t.mean(dim=-1, keepdim=True).squeeze().mean() for t in layer_norm_arr[:]]
        #     means = [i for i in means if not torch.isnan(i) and not torch.isinf(i) and i.item() > 0]
        #     mean_mean = torch.stack(means).mean() * loss_weight[2]
        #     loss += mean_mean



        return loss

    def get_true_label_and_y_tag(self, y, ids, second, pred):
        if self.cfg.model_config_num == 1:
            try:
                true_label = self.model(pixel_values=y.half()).pred_boxes
                y_tag = pred.pred_boxes
            except:
                true_label = self.model(y.half())
                y_tag = pred.logits
                true_label = true_label.logits
        elif self.cfg.model_config_num == 2:
            true_label = self.model(input_ids=ids[0], pixel_values=y).objectness_logits
            y_tag = pred.objectness_logits
        elif self.cfg.model_config_num == 4:
            true_label = self.model(input_ids=torch.randint(0, 100, (10, 10)), pixel_values=y).logits
            y_tag = pred.logits
        else:
            true_label = self.model(y).logits
            y_tag = pred.logits
        return true_label, y_tag

    def backward_and_cleanup(self, loss, x_grad):
        self.model.zero_grad()
        loss.backward()
        grads = x_grad.grad
        loss_val = loss.item()
        del loss, x_grad
        torch.cuda.empty_cache()
        return grads, loss_val

    # def loss_gradient(self, x, y, ids=None, loss_type="many_to_many"):
    #
    #     input_arr.clear()
    #     # save to image
    #     if loss_type == "universal":
    #         x_grad = x.clone().requires_grad_(True)
    #     else:
    #         x_grad = x.clone().detach().requires_grad_(True)
    #
    #     # del x  # delete the original tensor to free up memory
    #     second = False
    #     if self.cfg.model_config_num == 0:
    #
    #         pred = self.model(x_grad)
    #
    #         # for combine models
    #         # if random.random() < 0.5:
    #         #     pred = self.model(x_grad)
    #         #     second = False
    #         # else:
    #         #     pred = self.second_model(x_grad)
    #         #     second = True
    #     elif self.cfg.model_config_num == 1:
    #         pred = self.model(x_grad.half())  # for wisper model
    #     elif self.cfg.model_config_num == 2:
    #         pred = self.model(input_ids=ids[0], pixel_values=x_grad)  # for Owldetection model
    #     elif self.cfg.model_config_num == 3:  # for image captioning model
    #         gen_kwargs = {"max_length": 16, "num_beams": 4}
    #         pred = self.model.generate(x_grad, **gen_kwargs)
    #     elif self.cfg.model_config_num == 4:  # for image captioning model
    #         # pred = self.model.generate(pixel_values=x_grad, max_length=50)
    #         pred = self.model(input_ids=torch.randint(0, 100, (10, 10)), pixel_values=x_grad)
    #
    #     else:
    #         pred = self.model(x_grad)
    #
    #     matmul_lists = input_arr.copy()
    #
    #     self.iteration += 1
    #
    #     # Get the input and target tensors
    #
    #     inputs, targets = self.get_input_targeted(matmul_lists)
    #     # Count the number of outliers
    #     # total_outliers = sum([len(l) for l in outliers_arr])
    #     total_outliers, outs_ratio = count_outliers(outliers_arr_local,
    #                                                 threshold=self.cfg.model_threshold)  # compare with total_outliers
    #
    #     if self.iteration % 200 == 0 or self.iteration == 0:
    #         if hasattr(self.model.config, 'num_attention_heads'):
    #             blocks = self.model.config.num_hidden_layers
    #         else:
    #             blocks = self.model.config.text_config.num_hidden_layers
    #
    #         outliers_df = print_outliers(matmul_lists, outs_ratio, blocks)
    #         # print()
    #         # print(self.second_model.base_model_prefix if second else self.model.base_model_prefix)
    #         print()
    #         print(outliers_df)
    #     #
    #
    #     # Calculate the loss
    #     loss = torch.zeros(1, device="cuda")
    #     # add variance loss
    #
    #     for loss_fn, loss_weight in zip(self.loss_fns, self.loss_weights):
    #         # add loss for Linear8Bit
    #         for i in range(len(inputs)):
    #             temp_loss = loss_fn(inputs[i].to(torch.float64), targets[i].to(torch.float64)).squeeze().mean()
    #             loss.add_(loss_weight[0] * temp_loss)  # use in-place addition
    #             del temp_loss  # delete the temporary loss value
    #
    #         # add loss for accuracy
    #         if loss_weight[1] != 0:
    #             if second:
    #                 true_label = self.second_model(y).logits
    #                 y_tag = pred.logits
    #             if self.cfg.model_config_num == 1:
    #                 # true_label = self.model(input_ids=ids[1], pixel_values=y.half()).pred_boxes
    #                 try:
    #                     true_label = self.model(pixel_values=y.half()).pred_boxes
    #                     y_tag = pred.pred_boxes
    #                 except:
    #                     true_label = self.model(y.half())
    #                     y_tag = pred.logits
    #                     true_label = true_label.logits
    #             elif self.cfg.model_config_num == 2:
    #                 true_label = self.model(input_ids=ids[0], pixel_values=y).pred_boxes
    #                 y_tag = pred.pred_boxes
    #             elif self.cfg.model_config_num == 4:
    #
    #                 true_label = self.model(input_ids=torch.randint(0, 100, (10, 10)), pixel_values=y).logits
    #                 y_tag = pred.logits
    #             else:
    #                 true_label = self.model(y).logits
    #                 y_tag = pred.logits
    #
    #             loss += loss_weight[1] * loss_fn(y_tag, true_label).squeeze().mean()  # add accuracy loss
    #
    #         # add loss for total variation
    #         if loss_weight[2] != 0:
    #             c = self.tv(x_grad)
    #             loss += loss_weight[2] * c
    #
    #         # add loss for variance
    #         # variances = [torch.var(t[0], dim=(0, 1)).mean() for t in layer_norm_arr[:25]]
    #         # variances =  [i for i in variances if not torch.isinf(i)]
    #         # var_mean = torch.stack(variances).mean()
    #         # loss += var_mean * 50
    #     # Clear lists -
    #     clear_lists(input_arr, outliers_arr, outliers_arr_local, all_act, matmul_lists, layer_norm_arr)
    #
    #     if loss_type == "universal":
    #         return loss, total_outliers
    #
    #     self.model.zero_grad()
    #     loss.backward()
    #     grads = x_grad.grad
    #
    #     # Free up memory
    #     loss_val = loss.item()
    #     del loss
    #     del x_grad
    #     del pred
    #     torch.cuda.empty_cache()
    #
    #     return grads, loss_val, total_outliers

    # sponge attack loss

    # def loss_gradient(self, x, y, ids=None, loss_type="many_to_many"):
    #
    #     input_arr.clear()
    #
    #     x_grad = x.clone().detach().requires_grad_(True)
    #     pred = self.model(x_grad)
    #     matmul_lists = all_act.copy()
    #
    #     self.iteration += 1
    #     total_outliers, outs_ratio = count_outliers(outliers_arr_local,
    #                                                 threshold=self.cfg.model_threshold)  # compare with total_outliers
    #
    #     # if self.iteration % 200 == 0 or self.iteration == 0:
    #     #     if hasattr(self.model.config, 'num_attention_heads'):
    #     #         blocks = self.model.config.num_hidden_layers
    #     #     else:
    #     #         blocks = self.model.config.text_config.num_hidden_layers
    #     #
    #     #     outliers_df = print_outliers(matmul_lists, outs_ratio, blocks)
    #     #     # print()
    #     #     # print(self.second_model.base_model_prefix if second else self.model.base_model_prefix)
    #     #     print()
    #     #     print(outliers_df)
    #     #
    #
    #     # Calculate the loss
    #     loss = torch.zeros(1, device="cuda")
    #     # add  loss
    #     for i in range(len(matmul_lists)):
    #         loss += torch.norm(matmul_lists[i], p=2)
    #     loss = -loss
    #     # # Clear lists -
    #     clear_lists(input_arr, outliers_arr, outliers_arr_local, all_act, matmul_lists,layer_norm_arr)
    #     #
    #
    #
    #     self.model.zero_grad()
    #     loss.backward(retain_graph=True)
    #     grads = x_grad.grad
    #
    #     # Free up memory
    #     loss_val = loss.item()
    #     del loss
    #     del x_grad
    #     del pred
    #     torch.cuda.empty_cache()
    #
    #     return grads, loss_val, total_outliers

    # regular PGD
    # def loss_gradient(self, x, y, ids=None, loss_type="many_to_many"):
    #     input_arr.clear()
    #     if loss_type == "universal":
    #         x_grad = x.clone().requires_grad_(True)
    #     else:
    #         x_grad = x.clone().detach().requires_grad_(True)
    #     loss_fn = nn.CrossEntropyLoss()
    #
    #     # y_tag = self.model(x_grad)
    #     y_tag = self.model(x_grad).logits
    #
    #     total_outliers, outs_ratio = count_outliers(outliers_arr_local,
    #                                                 threshold=self.cfg.model_threshold)  # compare with total_outliers
    #     try:
    #         y = self.model(y).logits.argmax(dim=-1)
    #     except:
    #         y = self.model(y).argmax(dim=-1)
    #     # loss = loss_fn(y_tag, y)
    #     loss = -loss_fn(y_tag, y)
    #
    #
    #
    #     clear_lists(input_arr, outliers_arr, outliers_arr_local, all_act, layer_norm_arr)
    #
    #
    #     torch.cuda.empty_cache()
    #     if loss_type == "universal":
    #         return loss, total_outliers
    #
    #     self.model.zero_grad()
    #     loss.backward()
    #     grads = x_grad.grad
    #
    #     loss_val = loss.item()
    #     del loss
    #     del x_grad
    #     return grads, loss_val, total_outliers
    #
    #     def loss_gradient(self, x, y, ids=None, loss_type="many_to_many"):
    #     input_arr.clear()
    #     if loss_type == "universal":
    #         x_grad = x.clone().requires_grad_(True)
    #     else:
    #         x_grad = x.clone().detach().requires_grad_(True)
    #     loss_fn = nn.CrossEntropyLoss()
    #
    #     y_tag = self.model(x_grad)
    #     # y_tag = self.model(x_grad).logits
    #
    #     total_outliers, outs_ratio = count_outliers(outliers_arr_local,
    #                                                 threshold=self.cfg.model_threshold)  # compare with total_outliers
    #     try:
    #         y = self.model(y).logits.argmax(dim=-1)
    #     except:
    #         y = self.model(y).argmax(dim=-1)
    #     # loss = loss_fn(y_tag, y)
    #     loss = -loss_fn(y_tag, y)
    #
    #
    #
    #     clear_lists(input_arr, outliers_arr, outliers_arr_local, all_act, layer_norm_arr)
    #
    #
    #     torch.cuda.empty_cache()
    #     if loss_type == "universal":
    #         return loss, total_outliers
    #
    #     self.model.zero_grad()
    #     loss.backward()
    #     grads = x_grad.grad
    #
    #     loss_val = loss.item()
    #     del loss
    #     del x_grad
    #     return grads, loss_val, total_outliers
