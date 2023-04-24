import pandas as pd
import torch
from torchvision import transforms
import os
from pathlib import Path
from main_ViT import hook_fn, input_arr, outliers_arr, outliers_arr_local
# from transformers.models.vit.modeling_vit import before, after
from utils.general import save_graph, print_outliers
from utils.losses_utils import apply_weights

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import random
import seaborn as sns
from utils.attack_utils import count_outliers

class Loss:

    def __init__(self, model, loss_fns, convert_fn, attack_type, max_iter, images_save_path=None, mask=None,
                 weights=None,
                 **kwargs) -> None:
        super().__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fns = loss_fns
        self.convert_fn = convert_fn
        self.images_save_path = images_save_path
        self.mask = mask
        self.iteration = 0
        self.max_iter = max_iter
        self.attack_type = attack_type
        if self.images_save_path is not None:
            Path(self.images_save_path).mkdir(parents=True, exist_ok=True)
        if weights is not None:
            self.loss_weights = weights
        else:
            self.loss_weights = [1] * len(loss_fns)

    @staticmethod
    def get_topk_max_values(list1, list2, choice=0, k=1):

        if choice == 0:
            list1_max = list1.max(dim=2)[0]
            list2_max = list2.max(dim=2)[0]

        elif choice == 1:
            list1_max = torch.topk(list1.max(dim=2)[0], k=k)[0]
            topk_values, _ = torch.topk(list2.view(-1, 3072), k=k, dim=0)
            list2_max = topk_values.view(k, 3072)

        elif choice == 2:
            _, max_indices = torch.topk(list1, k=k, dim=2)
            list1_max = torch.gather(list1, 2, max_indices)
            _, max_indices = torch.topk(list2, k=k, dim=2)
            list2_max = torch.gather(list2, 2, max_indices)

        elif choice == 3:
            list1_max = list1.topk(k, dim=2)[0]
            list2_max = list2.topk(k, dim=2)[0]
        else:
            raise ValueError("Invalid choice. Choose between 1, 2 or 3.")

        return list1_max, list2_max

    @staticmethod
    def get_input_targeted(matmul_lists, iter):
        batch = matmul_lists[0].shape[0]

        matmul_lists = apply_weights(matmul_lists)

        # Stack list to tensor
        list1 = torch.stack([tensor for tensor in matmul_lists if tensor.size() == (batch, 197, 768)])
        list2 = torch.stack([tensor for tensor in matmul_lists if tensor.size() == (batch, 197, 3072)])


        # Create a Boolean mask that selects values under the threshold
        threshold = 7
        mask1 = list1_max < threshold
        mask2 = list2_max < threshold

        # Apply the mask to select the relevant values
        selected_values1 = list1_max[mask1]
        selected_values2 = list2_max[mask2]

        target1 = torch.full_like(selected_values1, 6)
        target2 = torch.full_like(selected_values2, 6)

        return selected_values1, selected_values2, target1, target2

    def loss_gradient(self, x, y):
        input_arr.clear()
        x_grad = x.clone().detach().requires_grad_(True)
        pred = self.model(x_grad)
        matmul_lists = input_arr.copy()

        # Save the image

        self.iteration += 1
        list1_max, list2_max, target1, target2 = self.get_input_targeted(matmul_lists, self.iteration)
        total_outliers = sum([len(t) for t in outliers_arr])
        local_total_outliers = count_outliers(outliers_arr_local, threshold=6)
        assert total_outliers == local_total_outliers

        if self.attack_type == 'OneToOneAttack':
            ex = "ex40"
            title = "max from layer column -> list1.max(dim=2)[0] list2_max = list2.max(dim=2)[0][:9]"
            save_graph(matmul_lists, outliers_arr, self.iteration, ex, title, total_outliers, self.max_iter)
            # save_image(x[0], f"/sise/home/barasa/8_bit/images_changes/{self.iteration}.jpg")
        else:
            if self.iteration == self.max_iter:
                print()
                print_outliers(matmul_lists, outliers_arr)
                self.iteration = 0

        input_arr.clear()
        outliers_arr.clear()

        loss = torch.zeros(1, device="cuda")
        for loss_fn, loss_weight in zip(self.loss_fns, self.loss_weights):
            loss += loss_weight * loss_fn(list1_max, target1).squeeze().mean()
            loss += loss_weight * loss_fn(list2_max, target2).squeeze().mean()
            # loss += torch.mean(-list2_max) #different loss function
            # loss += torch.mean(-list1_max) #different loss function

        self.model.zero_grad()
        loss.backward()
        grads = x_grad.grad
        return grads, loss.item(), total_outliers
