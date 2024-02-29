import torch
import torchvision
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

        self.LabeList = None
        self.Imglist = os.listdir(os.path.join(root, "images", split))
        if "coco" in root:
            self.LabeList = sorted([os.path.join(root, "labels", f"val_names",s) for s in os.listdir(os.path.join(root, "labels", f"val_names"))])
            self.Imglist = sorted(self.Imglist)
        self.ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)



    def __len__(self):
        return len(self.Imglist)

    def get_image_dir(self, index):
        try:
            return os.path.join(self.root, "images", self.split + "/") + self.Imglist[index]
        except:
            return os.path.join(self.root, "images", self.split + "/") + self.Imglist[0]

    def __getitem__(self, index):


        img_dir = self.get_image_dir(index)
        img = Image.open(img_dir)
        ids = 0

        try:
            is_whisper = "Whisper" in self.feature_extractor.feature_extractor_type
        except AttributeError:
            is_whisper = False

        if is_whisper:
            sample = next(iter(self.ds))
            img_extractor = self.feature_extractor(
                sample["audio"]["array"],
                sampling_rate=sample["audio"]["sampling_rate"],
                return_tensors="pt"
            ).input_features
            return img_extractor, "from dataset", ids
        is_owl = False
        try:
            is_owl = "Owl" in self.feature_extractor.image_processor_class or "Owl" in self.feature_extractor.feature_extractor_class
        except:
            pass


        try:
            if is_owl:
                with open(self.LabeList[index], 'r') as file:
                    text = file.readlines()
                    text = [item.strip() for item in text]
                texts = [text]
                feature_output = self.feature_extractor(text=texts, images=img, return_tensors="pt")
                img_extractor = feature_output["pixel_values"]
                ids = feature_output["input_ids"]
            else:
                img_extractor = self.feature_extractor(images=img, return_tensors="pt")["pixel_values"]
                # img_extractor = self.transforms(img)

        except Exception:
            img_dir = self.get_image_dir(0)
            img = Image.open(img_dir)
            img_extractor = self.feature_extractor(images=img, return_tensors="pt")["pixel_values"]

        return img_extractor, img_dir, ids


def get_dataset(dataset_name):
    # load dataset
    if dataset_name == 'imagenet':
        return ImageNetDataset
    if dataset_name == "coco":
        return ImageNetDataset



def get_loaders(loader_params, dataset_config, splits_to_load, model_name):
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
                                  shuffle=False,
                                  pin_memory=True)

    if 'validation' in splits_to_load:
        val_data = dataset(root=dataset_config['root_path'],
                           split='val', feature_extractor=model_feature_extractor)
        val_loader = DataLoader(val_data,
                                batch_size=loader_params['batch_size'],
                                num_workers=loader_params['num_workers'] ,
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

    # if train_loader is not None and val_loader is not None:
    #     train_data = ConcatDataset([train_data, val_data])
    #
    #     train_loader = DataLoader(train_data,
    #                               batch_size=loader_params['batch_size'],
    #                               num_workers=loader_params['num_workers'],
    #                               shuffle=True,
    #                               pin_memory=True)

    return train_loader, val_loader, test_loader
