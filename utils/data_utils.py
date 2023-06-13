import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset

import os

from PIL import Image
from transformers import ViTFeatureExtractor
from utils.model_utils import get_model, get_model_feature_extractor


class ImageNetDataset(Dataset):
    def __init__(self, root, split="val", feature_extractor=None, img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.n_classes = 1000
        self.feature_extractor = feature_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Imglist = os.listdir(os.path.join(root, "images", split))
        self.ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)

        if split != "test":
            self.LabeList = os.listdir(os.path.join(root, "labels", split))
        else:
            self.LabeList = None

    def __len__(self):
        return len(self.Imglist)

    def __getitem__(self, index):
        # read images and extract features using ViT
        img_dir = os.path.join(self.root, "images", self.split + "/") + self.Imglist[index]
        img = Image.open(img_dir)
        ids = None

        try:
            flag = "Whisper" in self.feature_extractor.feature_extractor_type
        except:
            flag = False

        if flag:
            sample = next(iter(self.ds))
            img_dir = "from dataset"
            img_extractor = self.feature_extractor(
                sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
            ).input_features

        try:
            flag = "Owl" in self.feature_extractor.feature_extractor_class
        except:
            flag = False


        else:
            try:
                if flag:
                    texts = [["cat", "dog", "animal","water",], ["car", "vehicle", "automobile"], ["person"]]
                    img_extractor = self.feature_extractor(text=texts, images=img, return_tensors="pt")["pixel_values"]
                    ids = self.feature_extractor(text=texts, images=img, return_tensors="pt")["input_ids"]
                else:
                    img_extractor = self.feature_extractor(images=img, return_tensors="pt")["pixel_values"]
            except:
                img = Image.open(os.path.join(self.root, "images", self.split + "/") + self.Imglist[index - 1])
                img_extractor = self.feature_extractor(images=img, return_tensors="pt")["pixel_values"]
            # read labels
            if self.LabeList is not None:
                with open(os.path.join(self.root, "labels", self.split + "/" + self.LabeList[index]), 'r') as file:
                    # Read all lines of the file into a list
                    lines = file.readlines()
                label = lines[0].split()[0]

        return img_extractor, img_dir,ids


def get_dataset(dataset_name):
    # load dataset
    if dataset_name == 'imagenet':
        return ImageNetDataset


def get_loaders(loader_params, dataset_config, splits_to_load, model_name, **kwargs):
    dataset_name = dataset_config['dataset_name']
    print('Loading {} dataset...'.format(dataset_name))
    train_loader, val_loader, test_loader = None, None, None
    model_feature_extractor = get_model_feature_extractor(model_name)
    dataset = get_dataset(dataset_name)
    if 'train' in splits_to_load:
        train_data = dataset(root=dataset_config['root_path'],
                             split='train', feature_extractor=model_feature_extractor)
        train_loader = DataLoader(train_data,
                                  batch_size=loader_params['batch_size'],
                                  num_workers=loader_params['num_workers'],
                                  shuffle=True,
                                  pin_memory=True)

    if 'validation' in splits_to_load:
        val_data = dataset(root=dataset_config['root_path'],
                           split='val')
        val_loader = DataLoader(val_data,
                                batch_size=loader_params['batch_size'],
                                num_workers=loader_params['num_workers'] // 2,
                                shuffle=False,
                                pin_memory=True)
    if 'test' in splits_to_load:
        test_data = dataset(root=dataset_config['root_path'],
                            split='test', feature_extractor=model_feature_extractor)
        test_loader = DataLoader(test_data,
                                 batch_size=loader_params['batch_size'],
                                 num_workers=loader_params['num_workers'] // 2,
                                 shuffle=False,
                                 pin_memory=True)
    if train_loader is not None and val_loader is not None:
        train_data = ConcatDataset([train_data, val_data])

        train_loader = DataLoader(train_data,
                                  batch_size=loader_params['batch_size'],
                                  num_workers=loader_params['num_workers'],
                                  shuffle=True,
                                  pin_memory=True)

    return train_loader, val_loader, test_loader
