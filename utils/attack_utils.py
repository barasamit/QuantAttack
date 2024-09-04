import torch


def count_outliers(outliers_arr, threshold=6):
    if len(outliers_arr) == 0:
        return 0,0
    batch_size = outliers_arr[0].shape[0]
    num_columns_with_outliers = 0
    outs_ration = []
    defence_ratio = 1

    for tensor in outliers_arr:
        # find elements with absolute value greater than threshold
        tensor = tensor.flatten(0,1)
        max_outliers = tensor.shape[-1] * defence_ratio
        abs_tensor = torch.abs(tensor)
        column_mask = torch.any(torch.ge(abs_tensor, threshold), dim=0)
        number_of_outliers_columns = torch.sum(column_mask).item() if torch.sum(column_mask).item() < max_outliers else max_outliers
        outs_ration.append(number_of_outliers_columns)
        num_columns_with_outliers += number_of_outliers_columns
    return num_columns_with_outliers ,outs_ration
    # return num_columns_with_outliers


