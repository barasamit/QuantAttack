import random
import numpy as np
import pandas as pd
import torch
import importlib
import os
import json
import pathlib
import inspect
import cv2
from sklearn import metrics


def print_data_frame(y):
    data = np.array(y).reshape(12, 6)
    df = pd.DataFrame(data)
    df.index = ["Block " + str(i) for i in range(1, 13)]
    df.columns = ["Layer " + str(i) for i in range(1, 7)]
    print(df)


def print_outliers(matmul_lists, outliers_arr):
    y = []
    for i, t in enumerate(matmul_lists):
        if t.size()[2] == 768:
            try:
                y.append((len(outliers_arr[i]) / 768) * 100)
            except:
                print("Error")
        else:
            y.append((len(outliers_arr[i]) / 3072) * 100)
    print_data_frame(y)


def save_graph(matmul_lists, outliers_arr, iteration,max_iter, ex=None, title=None, total_outliers=None):
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


def preplot(image):
    image = image * 255
    image = np.transpose(image, (1, 2, 0))
    out_image = np.flipud(np.clip(image, 0, 255))
    return out_image[60:, 62:-60, :]


def process_imgs(imgs, x1, x2, y1, y2):
    imgs = torch.flip(imgs, [2, 3])
    imgs = imgs[:, [2, 1, 0]]
    return crop_images(imgs, x1, x2, y1, y2)


def crop_images(images, x1, x2, y1, y2):
    return images[:, :, x1:x2, y1:y2]


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def load_npy_image_to_tensor(image_path, device):
    image_np = np.load(image_path).transpose((2, 0, 1))
    image_t = torch.from_numpy(image_np).unsqueeze(0)
    image_t = image_t.to(device)
    return image_t


def get_instance(module_name, instance_name):
    module = importlib.import_module(module_name)
    obj = getattr(module, instance_name)
    return obj


def save_class_to_file(config, current_folder):
    with open(os.path.join(current_folder, 'config.json'), 'w') as config_file:
        d = dict(vars(config))
        d_new = check_dict(d)
        json.dump(d_new, config_file)


def check_dict(d):
    d_new = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = check_dict(value)
        if isinstance(value, list):
            for i, val in enumerate(value):
                if isinstance(val, dict):
                    value[i] = check_dict(val)
        elif isinstance(value, (pathlib.PosixPath, torch.device)):
            value = str(value)
        elif isinstance(value, torch.Tensor):
            continue
        elif inspect.ismethod(value):
            value = value.__qualname__
        d_new[key] = value
    return d_new


def imgs_resize(imgs, resize_scale_height=256, resize_scale_width=256, keep_graph=True):
    outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], resize_scale_height, resize_scale_width)
    imgs = imgs.permute(0, 2, 3, 1)
    for i in range(imgs.size()[0]):
        if keep_graph:
            img = cv2.resize(src=imgs[i].numpy(), dsize=[resize_scale_height, resize_scale_width],
                             interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.resize(src=imgs[i].detach().cpu().numpy(), dsize=[resize_scale_width, resize_scale_height],
                             interpolation=cv2.INTER_CUBIC)
        outputs[i] = torch.FloatTensor(img.transpose(2, 0, 1).astype(np.float32))

    return outputs


def transform_for_save(image):
    if not isinstance(image, (np.ndarray, np.generic)):
        image = image.cpu().data.numpy()
    im_transpose = image.transpose(1, 2, 0)
    clip = np.clip(im_transpose, 0, 1)
    return clip


def auroc_aupr_scores(gts, preds, average_types):
    auc_dict = {}
    for average_type in average_types:
        auc = metrics.roc_auc_score(gts, preds, average=average_type)
        auc_dict[average_type] = auc

    return auc_dict
