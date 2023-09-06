import random
import numpy as np
import pandas as pd
import torch
import importlib
import os


def print_data_frame(y,blocks):
    number_of_blocks = blocks
    for i in range(0, len(y)):
        if len(y) % number_of_blocks != 0:
            y = y[0:-1+i]
        else:
            break

    number_of_layers = len(y) // number_of_blocks
    data = np.array(y).reshape(number_of_blocks, number_of_layers)
    df = pd.DataFrame(data)
    df.index = ["Block " + str(i) for i in range(1, number_of_blocks + 1)]
    df.columns = ["Layer " + str(i) for i in range(1, number_of_layers + 1)]
    return df
    # print(df)


# def print_outliers(matmul_lists, outliers_arr):
#     y = []
#     for i, t in enumerate(matmul_lists):
#         if t.size()[2] == 768:
#             try:
#                 y.append((len(outliers_arr[i]) / 768) * 100)
#             except:
#                 print("Error")
#         else:
#             y.append((len(outliers_arr[i]) / 3072) * 100)
#     return print_data_frame(y)
def print_outliers(matmul_lists, outliers_arr,blocks):
    y = []
    for i, t in enumerate(matmul_lists):

        try:
            size = t.size()[2]  # Get the size dynamically
            y.append((outliers_arr[i] / size) * 100)
        except:
            print("Error")
    return print_data_frame(y,blocks)



def save_graph(matmul_lists, outliers_arr, iteration, max_iter, ex=None, title=None, total_outliers=None):
    y = []
    for i, t in enumerate(matmul_lists):
        if t.size()[2] == 768:
            try:
                y.append((len(outliers_arr[i]) / 768) * 100)
            except:
                print("Error")
        else:
            y.append((len(outliers_arr[i]) / 3072) * 100)
    # save bar plot
    x = [i for i in range(len(y))]

    if iteration == max_iter or iteration % 1000 == 0:
        dir_path = f"/sise/home/barasa/8_bit/grid_search/{ex}/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory {dir_path} created.")
            name = os.path.join(dir_path, f"ex_list_{iteration}.txt")
            name_title = os.path.join(dir_path, f"ex_title.txt")
            name_outliers = os.path.join(dir_path, f"outliers_number.txt")

            with open(name, 'w') as f:
                for item1, item2 in zip(x, y):
                    f.write(f'{item1}\t{item2}\n')
            with open(name_title, 'w') as f:
                f.write(f'{title}')
            with open(name_outliers, 'w') as f:
                f.write(f'{total_outliers}')

            print_data_frame(y)

        else:
            print_data_frame(y)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def get_instance(module_name, instance_name):
    module = importlib.import_module(module_name)
    obj = getattr(module, instance_name)
    return obj


def get_patch(config):
    if config.initial_patch == 'random':
        patch = torch.rand(
            (1, 3, config.image_size,  config.image_size),dtype=torch.float32)
    elif config.initial_patch == 'ones':
        patch = torch.ones(
            (1, 3, config.image_size,  config.image_size),dtype=torch.float32)
    elif config.initial_patch == 'zeros':
        patch = torch.zeros(
            (1, 3, config.image_size,  config.image_size),dtype=torch.float32)
    elif config.initial_patch == 'half':
        patch = torch.full(
            (1, 3, config.image_size,  config.image_size),dtype=torch.float32)

    # patch = patch.to(config.device)
    patch = patch.requires_grad_(True)
    return patch