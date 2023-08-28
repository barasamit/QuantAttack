from pathlib import Path

import torch

from utils.attack_utils import count_outliers
from utils.general import save_graph, print_outliers
from utils.init_collect_arrays import input_arr, outliers_arr, outliers_arr_local, pointers
from utils.losses_utils import clear_lists, stack_tensors_with_same_shape
import torch.nn as nn
from torchvision.utils import save_image
from torchmetrics.image import TotalVariation


class ImageData:
    def __init__(self):
        self._pixel_values = None

    def pixel_values(self):
        return self._pixel_values


class Loss:

    def __init__(self, model, loss_fns, convert_fn, cfg, images_save_path=None, mask=None,
                 weights=None,
                 **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fns = loss_fns
        self.convert_fn = convert_fn
        self.images_save_path = images_save_path
        self.mask = mask
        self.tv = TotalVariation().to(self.device)
        self.iteration = 0
        self.max_iter = cfg.attack_params["max_iter"]
        self.attack_type = cfg.attack_type
        if self.images_save_path is not None:
            Path(self.images_save_path).mkdir(parents=True, exist_ok=True)
        if weights is not None:
            self.loss_weights = weights
        else:
            self.loss_weights = [1] * len(loss_fns)

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
            t_max = tensor.topk(self.cfg.num_topk_values, dim=2)[0]
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

    def denormalize(self,x):
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
        # save to image
        if loss_type == "universal":
            x_grad = x.clone().requires_grad_(True)
        else:
            x_grad = x.clone().detach().requires_grad_(True)

        # del x  # delete the original tensor to free up memory

        if self.cfg.model_config_num == 0:
            pred = self.model(x_grad)
        elif self.cfg.model_config_num == 1:
            pred = self.model(x_grad.half())  # for wisper model
        elif self.cfg.model_config_num == 2:
            pred = self.model(input_ids=ids[0], pixel_values=x_grad)  # for Owldetection model
        else:
            pred = self.model(x_grad)

        # matmul_lists = filter_items_by_pointer(input_arr.copy(), pointers.copy())

        matmul_lists = input_arr.copy()
        self.iteration += 1

        # Get the input and target tensors
        inputs, targets = self.get_input_targeted(matmul_lists)

        # Count the number of outliers
        # total_outliers = sum([len(l) for l in outliers_arr])
        total_outliers = count_outliers(outliers_arr_local,
                                        threshold=self.cfg.model_threshold)  # compare with total_outliers

        if self.iteration % 200 == 0 or self.iteration == 0:
            if hasattr(self.model.config, 'num_attention_heads'):
                blocks = self.model.config.num_attention_heads
            else:
                blocks = self.model.config.text_config.num_hidden_layers

            outliers_df = print_outliers(matmul_lists, outliers_arr, blocks)
            print()
            print(outliers_df)


        # if self.iteration % 100 == 0:
        #     save_image(self.denormalize(x.clone()), f"im_change_2/outliers {total_outliers}.png")



        # Calculate the loss
        loss = torch.zeros(1, device="cuda")

        for loss_fn, loss_weight in zip(self.loss_fns, self.loss_weights):
            # add loss for Linear8Bit
            for i in range(len(inputs)):
                temp_loss = loss_fn(inputs[i].to(torch.float64), targets[i].to(torch.float64)).squeeze().mean()
                loss.add_(loss_weight[0] * temp_loss)  # use in-place addition
                del temp_loss  # delete the temporary loss value

            # add loss for accuracy
            if loss_weight[1] != 0:
                true_label = self.model(y).logits
                loss += loss_weight[1] * loss_fn(pred.logits, true_label).squeeze().mean()  # add accuracy loss

            if loss_weight[2] != 0:
                # add loss for total variation

                c = self.tv(x_grad)
                loss += loss_weight[2] * c

        # Clear lists -
        clear_lists(input_arr, outliers_arr, outliers_arr_local, pointers, matmul_lists)

        if loss_type == "universal":
            return loss, total_outliers

        self.model.zero_grad()
        loss.backward()
        grads = x_grad.grad

        # Free up memory
        loss_val = loss.item()
        del loss
        del x_grad
        del pred
        torch.cuda.empty_cache()

        return grads, loss_val, total_outliers
