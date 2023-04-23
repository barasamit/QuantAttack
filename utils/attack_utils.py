import torch


def count_outliers(outliers_arr, threshold):
    outliers_count = 0
    all_shapes = list(set([value.shape for value in outliers_arr]))  # get all unique shapes from list
    for shape in all_shapes:
        # stack all inputs of the same shape together
        stacked_linear = torch.stack(list(filter(lambda x: x.shape == shape, outliers_arr)))
        # get highest value in each column
        highest_stacked_linear = stacked_linear.max(dim=-2)[0]
        # count outliers
        outliers_count += (highest_stacked_linear >= threshold).sum()
    return outliers_count
