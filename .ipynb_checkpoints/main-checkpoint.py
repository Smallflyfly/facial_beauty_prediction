# -*- coding: utf-8 -*-
import argparse
import os

import torch
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from flyai.utils import remote_helper
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset import FacialBeautyDataset
from model.osnet import osnet_x1_0
from path import MODEL_PATH
from utils.utils import load_pretrained_weights, build_optimizer, build_scheduler
import torch.nn as nn


'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("FacialBeautyPrediction")

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        pass

    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        dataset = FacialBeautyDataset()
        model = osnet_x1_0(num_classes=1, pretrained=True, loss='smoothL1Loss', use_gpu=True)
        # load_pretrained_weights(model, './weights/pretrained/osnet_x1_0_imagenet.pth')
        path = remote_helper.get_remote_data('https://www.flyai.com/m/osnet_x1_0_imagenet.pth')
        load_pretrained_weights(model, path)
        model = model.cuda()
        optimizer = build_optimizer(model)
        max_epoch = args.EPOCHS
        batch_size = args.BATCH
        scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=max_epoch)
        criterion = nn.SmoothL1Loss()
        model.train()
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        cudnn.benchmark = True
        for epoch in range(max_epoch):
            for index, data in enumerate(train_loader):
                im, label = data
                im = im.cuda()
                label = label.float().cuda()
                optimizer.zero_grad()
                out = model(im)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                if index % 100 == 0:
                    print("Epoch: [{}/{}][{}/{}]  Loss {:.4f}".format(epoch+1, max_epoch, index+1,
                                                                                  len(train_loader), loss))
            scheduler.step()
        torch.save(model.state_dict(), 'last.pth')


if __name__ == '__main__':
    main = Main()
    # main.download_data()
    main.train()
