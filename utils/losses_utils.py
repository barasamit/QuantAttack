import torch
# import lpips


class MSE:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        # self.func = torch.nn.MSELoss(**kwargs)
        self.func = torch.nn.MSELoss()

    def __call__(self, x, y):
        return self.func(x, y)


class L1:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.func = torch.nn.L1Loss(**kwargs)

    def __call__(self, x, y):
        return self.func(x, y)


