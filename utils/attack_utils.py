import torch


def count_outliers(outliers_arr, threshold):
    batch_size = outliers_arr[0].shape[0]
    num_columns_with_outliers = 0
    for tensor in outliers_arr:
        # find elements with absolute value greater than threshold
        abs_tensor = torch.abs(tensor)
        column_mask = torch.any(torch.ge(abs_tensor, threshold), dim=1)
        num_columns_with_outliers += torch.sum(column_mask).item()
    return num_columns_with_outliers // batch_size
