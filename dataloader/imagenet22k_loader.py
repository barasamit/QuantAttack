import os
import random


import glob

import albumentations as A
import cv2
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils import data
from tqdm import tqdm
from yolov5.utils.augmentations import letterbox, random_perspective


class imageNet22Kloader( data.Dataset ):
    def __init__(self, hyp, root='./datasets/pascal/', split="voc12-train", img_transform=None, label_transform=None,
                 augment=True, img_size=224):
        self.root = root
        self.split = split
        self.n_classes = 20
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_size = img_size

        # if split == "test":
        #     self.GT = None
        # else:
        #     # filePath = self.root + self.split + '.mat'
        #     # datafile = loadmat( filePath )
        #     # self.GT = datafile['labels']

        self.Imglist = []
        dirs = [f.path for f in os.scandir( root ) if f.is_dir()]
        for dir in tqdm(dirs):
            temp_list = [dir + "/" + s for s in os.listdir( dir )]
            self.Imglist += temp_list



        self.hyp = hyp
        self.augment = augment
        self.transform = A.Compose( [
            A.Blur( p=0.01 ),
            A.MedianBlur( p=0.01 ),
            A.ToGray( p=0.01 ),
            A.CLAHE( p=0.01 ),
            A.RandomBrightnessContrast( p=0.0 ),
            A.RandomGamma( p=0.0 ),
            A.ImageCompression( quality_lower=75, p=0.0 )] )

    def __len__(self):
        return len( self.Imglist )

    def __getitem__(self, index):
        img = cv2.imread( self.Imglist[index].strip() )
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max( h0, w0 )  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize( img, (int( w0 * r ), int( h0 * r )),
                              interpolation=cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA )

        img, ratio, pad = letterbox( img, self.img_size, auto=False, scaleup=False )

        img = img.transpose( (2, 0, 1) )[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray( img )

        return torch.from_numpy( img / 255 ).float(), torch.tensor( [-1] )
