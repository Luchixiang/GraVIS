from __future__ import print_function

import numpy as np
from skimage import color

import torch
import torchvision.datasets as datasets
import os
import pandas as pd
from PIL import Image


class GraVISDataset(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, imgs, transform=None, target_transform=None, two_crop=False):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.two_crop = two_crop

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        # path, target = self.imgs[index]
        out = []
        path = self.imgs[index]
        # image = self.loader(path)
        image = Image.open(path).convert('RGB')
        for i in range(2):
            img = self.transform(image)
            out.append(img)
        return torch.stack(out, dim=0)


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2HSV(object):
    """Convert RGB PIL image to ndarray HSV."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hsv(img)
        return img


class RGB2HED(object):
    """Convert RGB PIL image to ndarray HED."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hed(img)
        return img


class RGB2LUV(object):
    """Convert RGB PIL image to ndarray LUV."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2luv(img)
        return img


class RGB2YUV(object):
    """Convert RGB PIL image to ndarray YUV."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yuv(img)
        return img


class RGB2XYZ(object):
    """Convert RGB PIL image to ndarray XYZ."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2xyz(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to ndarray YIQ."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class RGB2CIERGB(object):
    """Convert RGB PIL image to ndarray RGBCIE."""

    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2rgbcie(img)
        return img
