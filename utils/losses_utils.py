import torch



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


def get_blocks(matmul_lists):
    b1 = matmul_lists[0:6]
    b2 = matmul_lists[6:12]
    b3 = matmul_lists[12:18]
    b4 = matmul_lists[18:24]
    b5 = matmul_lists[24:30]
    b6 = matmul_lists[30:36]
    b7 = matmul_lists[36:42]
    b8 = matmul_lists[42:48]
    b9 = matmul_lists[48:54]
    b10 = matmul_lists[54:60]
    b11 = matmul_lists[60:66]
    b12 = matmul_lists[66:72]

    return b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12



    # Get blocks
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12 = get_blocks(matmul_lists)

    # Get weights

    # Apply weights
    mul_tensors_by_scalar = lambda tensor_list, scalar: [tensor.mul(scalar) for tensor in tensor_list]

    # function with downscale
    # mul_tensors_by_scalar = lambda tensor_list, scalar: [tensor.mul(scalar - i) for i, tensor in enumerate(tensor_list)]

    tensor_with_w8 = mul_tensors_by_scalar(b1, w1) + mul_tensors_by_scalar(b2, w2) + mul_tensors_by_scalar(b3,
                                                                                                           w3) + mul_tensors_by_scalar(
        b4, w4) + mul_tensors_by_scalar(b5, w5) + mul_tensors_by_scalar(b6, w6) + mul_tensors_by_scalar(b7,
                                                                                                        w7) + mul_tensors_by_scalar(
        b8, w8) + mul_tensors_by_scalar(b9, w9) + mul_tensors_by_scalar(b10, w10) + mul_tensors_by_scalar(b11,
                                                                                                          w11) + mul_tensors_by_scalar(
        b12, w12)
    return tensor_with_w8

