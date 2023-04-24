import torch


def count_outliers(outliers_arr, threshold):
    num_columns_with_outliers = 0
    for tensor in outliers_arr:
        # find columns with at least one element above the threshold
        column_mask = torch.any(torch.gt(tensor, threshold), dim=1)
        num_columns_with_outliers += torch.sum(column_mask).item()
    return num_columns_with_outliers
