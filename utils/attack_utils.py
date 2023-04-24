import torch


# def count_outliers(outliers_arr, threshold):
#     outliers_count = 0
#     all_shapes = list(set([value.shape for value in outliers_arr]))  # get all unique shapes from list
#     for shape in all_shapes:
#         # stack all inputs of the same shape together
#         stacked_linear = torch.stack(list(filter(lambda x: x.shape == shape, outliers_arr)))
#         # get highest value in each column
#         highest_stacked_linear = stacked_linear.max(dim=-2)[0]
#         # count outliers
#         outliers_count += (highest_stacked_linear >= threshold).sum()
#     return outliers_count


def count_outliers(outliers_arr, threshold):
    num_columns_with_outliers = 0
    for tensor in outliers_arr:
        # find columns with at least one element above the threshold
        column_mask = torch.any(torch.gt(tensor, threshold), dim=1)
        num_columns_with_outliers += torch.sum(column_mask).item()
    return num_columns_with_outliers


def count_outliers_list(outliers_arr, threshold):
    num_outliers_list = []
    for tensor in outliers_arr:
        # Convert tensor to PyTorch and find columns with at least one element above the threshold
        column_mask = torch.any(torch.gt(tensor, threshold), dim=1)
        num_outliers = torch.sum(column_mask).item()
        num_outliers_list.append(num_outliers)
    return num_outliers_list
