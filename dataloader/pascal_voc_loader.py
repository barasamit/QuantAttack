import os
from PIL import Image
from torch.utils import data


class pascalVOCLoader(data.Dataset):
    def __init__(self, root, split="val", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.n_classes = 20
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.Imglist = os.listdir(os.path.join(root, "images", split))
        self.lablist = os.listdir(os.path.join(root, "labels", split))

    def __len__(self):
        return len(self.Imglist)

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.root, "images", self.split+"/") + self.Imglist[index])

        if self.img_transform is not None:
            imgs = self.img_transform(img)
        else:
            imgs = img

        with open(os.path.join(self.root, "labels", self.split+"/" + self.lablist[index]), 'r') as file:
            # Read all lines of the file into a list
            lines = file.readlines()
        label = lines[0].split()[0]

        return imgs, label
