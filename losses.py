import torch
from pathlib import Path
from utils.init_collect_arrays import input_arr, outliers_arr, outliers_arr_local, pointers
from utils.general import save_graph, print_outliers
from utils.losses_utils import apply_weights, clear_lists, filter_items_by_pointer, stack_tensors_with_same_shape

from utils.attack_utils import count_outliers
import torch.nn as nn
import torch.nn.functional as F


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

    @staticmethod
    def get_topk_max_values(list1, list2, choice=0, k=1):

        if choice == 0:
            list1_max = list1.topk(k, dim=2)[0]
            list2_max = list2.topk(k, dim=2)[0]

        elif choice == 1:
            list1_max = torch.topk(list1.max(dim=2)[0], k=k)[0]
            topk_values, _ = torch.topk(list2.view(-1, 3072), k=k, dim=0)
            list2_max = list2.topk(k, dim=2)[0]

        elif choice == 2:
            _, max_indices = torch.topk(list1, k=k, dim=2)
            list1_max = torch.gather(list1, 2, max_indices)
            _, max_indices = torch.topk(list2, k=k, dim=2)
            list2_max = torch.gather(list2, 2, max_indices)

        else:
            raise ValueError("Invalid choice. Choose between 1, 2 or 3.")

        return list1_max, list2_max

    def get_input_targeted(self, matmul_lists):
        # Get the batch size, rows and columns -> if use other vit model, change the shape and maybe more shapes

        # Apply weights
        # lists_with_weights = apply_weights(matmul_lists, self.cfg)
        stacked_tensors = stack_tensors_with_same_shape(matmul_lists)

        # Get the top k values
        threshold = self.cfg.model_threshold_dest

        selected_values_list = []
        targets_list = []
        for tensor in stacked_tensors:

            tensor = tensor.abs()
            list1_max = tensor.topk(self.cfg.num_topk_values, dim=2)[0]
            mask_lower = list1_max > 2
            mask_upper = list1_max < threshold

            # Combine the two masks using the logical AND operator &
            mask = mask_lower & mask_upper
            # mask = mask_upper
            selected_values = list1_max[mask]
            if len(selected_values) == 0:
                selected_values = torch.tensor([self.cfg.target]).to(self.device)
            target = torch.full_like(selected_values, self.cfg.target)

            selected_values_list.append(selected_values)
            targets_list.append(target)

        return selected_values_list, targets_list



    def loss_gradient(self, x, y):
        input_arr.clear()
        x_grad = x.clone().detach().requires_grad_(True)
        pred = self.model(x_grad).logits

        # matmul_lists = filter_items_by_pointer(input_arr.copy(), pointers.copy())
        matmul_lists = input_arr.copy()
        self.iteration += 1

        # Get the input and target tensors
        # list1_max, list2_max, target1, target2 = self.get_input_targeted(matmul_lists)
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
                loss += loss_weight * loss_fn(inputs[i], targets[i]).squeeze().mean()
            # loss += loss_weight * loss_fn(list1_max, target1).squeeze().mean()
            # loss += loss_weight * loss_fn(list2_max, target2).squeeze().mean()

        # loss += 100 * nn.MSELoss(pred, true_label).squeeze().mean() # loss with accuracy

            # loss += torch.mean(-list2_max) #different loss function
            # loss += torch.mean(-list1_max) #different loss function

        self.model.zero_grad()
        loss.backward()
        grads = x_grad.grad
        return grads, loss.item(), total_outliers
