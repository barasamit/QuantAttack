from pathlib import Path

import torch

from utils.attack_utils import count_outliers
from utils.general import save_graph, print_outliers
from utils.init_collect_arrays import input_arr, outliers_arr, outliers_arr_local, pointers
from utils.losses_utils import clear_lists, stack_tensors_with_same_shape


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

        selected_values_list = []
        targets_list = []

        for i, tensor in enumerate(stacked_tensors):
            # attack specific layers
            # if i == 0:
            #     tensor = tensor[:, :42, :]
            # elif i == 1:
            #     tensor = tensor[:, :8, :]

            # tensor = tensor[:, :, 0:1, :]  # take only the first row

            tensor = tensor.abs()
            t_max = tensor.topk(self.cfg.num_topk_values, dim=2)[0]
            mask_lower = t_max > 2
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

        return selected_values_list, targets_list

    def loss_gradient(self, x, y):
        clear = input_arr.clear()
        x_grad = x.clone().detach().requires_grad_(True)
        pred = self.model(x_grad).logits

        # matmul_lists = filter_items_by_pointer(input_arr.copy(), pointers.copy())
        matmul_lists = input_arr.copy()
        self.iteration += 1

        # Get the input and target tensors
        inputs, targets = self.get_input_targeted(matmul_lists)

        # Count the number of outliers
        total_outliers = sum([len(l) for l in outliers_arr])
        local_total_outliers = count_outliers(outliers_arr_local,
                                              threshold=self.cfg.model_threshold)  # compare with total_outliers
        # assert total_outliers == local_total_outliers

        # Save the image
        if self.attack_type == 'OneToOneAttack':
            ex = "ex40"  # save only if name not exists
            title = "max from layer column -> list1.max(dim=2)[0] list2_max = list2.max(dim=2)[0][:9]"
            save_graph(matmul_lists, outliers_arr, self.iteration, self.max_iter, ex, title, total_outliers)
            # save_graph(matmul_lists, outliers_arr, iteration, max_iter, ex=None, title=None, total_outliers=None)
            # save_image(x[0], f"/sise/home/barasa/8_bit/images_changes/{self.iteration}.jpg")

        outliers_df = print_outliers(matmul_lists, outliers_arr)
        if self.iteration % 200 == 0:
            print()
            print(outliers_df)

        true_label = self.model(y).logits

        # Clear lists
        clear_lists(input_arr, outliers_arr, outliers_arr_local, pointers)

        # Calculate the loss
        loss = torch.zeros(1, device="cuda")
        for loss_fn, loss_weight in zip(self.loss_fns, self.loss_weights):
            for i in range(len(inputs)):
                loss += loss_weight * loss_fn(inputs[i].to(torch.float64),
                                              targets[i].to(torch.float64)).squeeze().mean()
            # loss += 50 * loss_fn(pred, true_label).squeeze().mean()   # add accuracy loss

        self.model.zero_grad()
        loss.backward()
        grads = x_grad.grad
        return grads, loss.item(), total_outliers
