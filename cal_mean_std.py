#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/03/24
"""
import os
import numpy as np
import cv2


def cal():
    path = "data/input/FacialBeautyPrediction/image"
    files = os.listdir(path)
    R_means = 0
    B_means = 0
    G_means = 0
    R_stds = 0
    B_stds = 0
    G_stds = 0
    n = len(files)
    for image in files:
        image = os.path.join(path, image)
        im = cv2.imread(image)
        im_R = im[:, :, 0] / 255
        im_B = im[:, :, 1] / 255
        im_G = im[:, :, 2] / 255
        # BGR
        R_mean = np.mean(im_R)
        B_mean = np.mean(im_B)
        G_mean = np.mean(im_G)
        R_std = np.std(im_R)
        B_std = np.std(im_B)
        G_std = np.std(im_G)
        R_means += R_mean
        B_means += B_mean
        G_means += G_mean
        R_stds += R_std
        B_stds += B_std
        G_stds += G_std
    R_means, G_means, B_means = R_means / n, G_means / n, B_means / n
    R_stds, G_stds, B_stds = R_stds / n, G_stds / n, B_stds / n
    print(R_means, G_means, B_means)
    print(R_stds, G_stds, B_stds)
    # 0.5682438590709008 0.683464602350031 0.5977400307759463
    # 0.32749810747882546 0.3024549688794429 0.3172456031142563


if __name__ == '__main__':
    cal()