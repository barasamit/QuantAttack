# inspired by adversarial-robustness-toolbox (IBM)

import torch
from tqdm import tqdm
import os


class ProjectedGradientDescent:
    def __init__(self,
                 loss_function=None,
                 norm="inf",
                 eps=0.3,
                 eps_step=0.1,
                 decay=None,
                 max_iter=100,
                 targeted=False,
                 num_random_init=1,
                 device='cpu',
                 clip_values=(0, 1)) -> None:
        super().__init__()
        self.loss_function = loss_function
        self.norm = norm
        self.decay = decay
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.device = device
        self.eps_step = torch.tensor(eps_step)
        self.eps = torch.tensor(eps, dtype=torch.float32, device=device)
        self.clip_min = torch.tensor(clip_values[0], dtype=torch.float32, device=self.device)
        self.clip_max = torch.tensor(clip_values[1], dtype=torch.float32, device=self.device)
        self.outliers_num = 0

    def generate(self, inputs, targets, batch_info):

        return self._generate_batch(inputs, targets, batch_info)

    def _generate_batch(self, inputs, targets, batch_info):
        adv_x = inputs.clone()
        momentum = torch.zeros(inputs.shape)
        progress_bar = tqdm(range(self.max_iter), total=self.max_iter, ncols=150,
                            desc='Batch {}/{} '.format(batch_info['cur'], batch_info['total']))
        self.loss_values = []
        for i, _ in enumerate(progress_bar):
            adv_x = self._compute(adv_x, inputs, targets, momentum)
            progress_bar.set_postfix_str(
                'Batch Loss: {:.4} , number of outliers {}'.format(self.loss_values[-1], self.outliers_num))

        return adv_x

    def _compute(self, x_adv, x_init, targets, momentum):
        # handle random init
        perturbation = self._compute_perturbation(x_adv, targets, momentum)
        x_adv = self._apply_perturbation(x_adv, perturbation)
        perturbation = self._projection(x_adv - x_init)
        x_adv = perturbation + x_init
        return x_adv

    def _compute_perturbation(self, adv_x, targets, momentum):
        tol = 10e-8
        grad, loss_value, self.outliers_num = self.loss_function(adv_x, targets)
        self.loss_values.append(loss_value)
        grad = grad * (1 - 2 * int(self.targeted))
        if torch.any(grad.isnan()):
            grad[grad.isnan()] = 0.0

        # Apply momentum
        if self.decay is not None:
            ind = tuple(range(1, len(adv_x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore
            grad = self.decay * momentum + grad
            # Accumulate the gradient for the next iter
            momentum += grad

        # Apply norm
        if self.norm == "inf":
            grad = grad.sign()
        elif self.norm == 1:
            ind = tuple(range(1, len(adv_x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdim=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(adv_x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, dim=ind, keepdim=True)) + tol)

        return grad

    def _apply_perturbation(self, adv_x, perturbation):
        perturbation_step = self.eps_step * perturbation
        perturbation_step[torch.isnan(perturbation_step)] = 0
        adv_x = adv_x + perturbation_step
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = torch.max(
                torch.min(adv_x, self.clip_max),
                self.clip_min,
            )
        return adv_x

    def _projection(self, values):
        tol = 10e-8
        values_tmp = values.reshape(values.shape[0], -1)
        if self.norm == 2:
            values_tmp = (values_tmp *
                          torch.min(
                              torch.tensor([1.0], dtype=torch.float32).to(self.device),
                              self.eps / (torch.norm(values_tmp, p=2, dim=1) + tol),
                          ).unsqueeze_(-1)
                          )
        elif self.norm == 1:
            values_tmp = (values_tmp *
                          torch.min(
                              torch.tensor([1.0], dtype=torch.float32).to(self.device),
                              self.eps / (torch.norm(values_tmp, p=1, dim=1) + tol),
                          ).unsqueeze_(-1)
                          )
        elif self.norm == 'inf':
            values_tmp = values_tmp.sign() * torch.min(values_tmp.abs(), self.eps)
        values = values_tmp.reshape(values.shape)
        return values
