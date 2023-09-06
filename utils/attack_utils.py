import torch


def count_outliers(outliers_arr, threshold):
    batch_size = outliers_arr[0].shape[0]
    num_columns_with_outliers = 0
    outs_ration = []
    for tensor in outliers_arr:
        # find elements with absolute value greater than threshold
        tensor = tensor.flatten(0,1)
        abs_tensor = torch.abs(tensor)
        column_mask = torch.any(torch.ge(abs_tensor, threshold), dim=0)
        outs_ration.append(torch.sum(column_mask).item())
        num_columns_with_outliers += torch.sum(column_mask).item()
    return num_columns_with_outliers ,outs_ration
    # return num_columns_with_outliers
