#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/03/24
"""
import os

import pandas
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from path import DATA_PATH


class FacialBeautyDataset(Dataset):
    def __init__(self):
        self.image_path = os.path.join(os.path.join(DATA_PATH, 'FacialBeautyPrediction'), 'image')
        self.train_image = []
        self.labels = []
        self.num_classes = 1
        self.label_map = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((360, 360)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.568, 0.683, 0.597], [0.327, 0.302, 0.317])
            ]
        )
        self.process_data()

    def process_data(self):
        images = os.listdir(self.image_path)
        self.read_csv()
        for image in images:
            if image.endswith(".jpg"):
                self.train_image.append(os.path.join(self.image_path, image))
                self.labels.append(self.label_map[image])

    def read_csv(self):
        csv = 'data/input/FacialBeautyPrediction/train.csv'
        data_frame = pandas.read_csv(csv)
        image_path = data_frame['image_path']
        label = data_frame['label']
        for image, label in zip(image_path, label):
            image = image.replace('./image/', '')
            self.label_map[image] = label

    def __getitem__(self, index):
        image = self.train_image[index]
        label = self.labels[index]
        im = Image.open(image)
        im = self.transform(im)
        # label [0, 5] -> [0, 1]
        label = label / 5
        return im, label

    def __len__(self):
        return len(self.train_image)