import torch


class MSE:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        # self.func = torch.nn.MSELoss(**kwargs)
        self.func = torch.nn.MSELoss()

    def __call__(self, x, y):
        return self.func(x, y)


class BCEWithLogitsLoss:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        # self.func = torch.nn.MSELoss(**kwargs)
        self.func = torch.nn.BCEWithLogitsLoss()

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


def apply_weights(matmul_lists, cfg):
    # Get blocks
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12 = get_blocks(matmul_lists)

    # Get weights
    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12 = cfg.blocks_weights

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


def clear_lists(*lists):
    for lst in lists:
        lst.clear()


def filter_items_by_pointer(items, pointers, mode="filter"):
    unique_items = {}
    if mode == "filter":
        for item, pointer in zip(items, pointers):
            unique_items[pointer] = item
    else:
        for item, pointer in zip(items, pointers):
            if pointer in unique_items:
                unique_items[pointer] += item
            else:
                unique_items[pointer] = item
    return list(unique_items.values())


def stack_tensors_with_same_shape(matmul_lists):
    # Create a dictionary to store tensors with the same shape
    tensor_dict = {}
    for tensor in matmul_lists:
        shape = tensor.shape
        if shape in tensor_dict:
            tensor_dict[shape].append(tensor)
        else:
            tensor_dict[shape] = [tensor]

    # Create a list to store the stacked tensors
    stacked_tensors = []

    # Iterate over the dictionary of tensors with the same shape
    for shape, tensor_list in tensor_dict.items():
        if len(tensor_list) == 1:
            stacked_tensors.append(tensor_list[0])
        else:
            stacked_tensor = torch.stack(tensor_list)
            stacked_tensors.append(stacked_tensor)

    # permute to get the right shape (batch, num_layers, rows, cols)
    stacked_tensors = [tensor.permute(1, 0, 2, 3) for tensor in stacked_tensors]

    return stacked_tensors


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
