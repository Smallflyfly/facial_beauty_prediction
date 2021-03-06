# -*- coding: utf-8 -*
from PIL import Image
from flyai.framework import FlyAI
from torch.backends import cudnn
from torchvision import transforms

from model.osnet import osnet_x1_0
from utils.utils import load_pretrained_weights


class Prediction(FlyAI):
    def __init__(self):
        self.model = self.load_model()
        self.model = self.model.cuda()
        self.transform = transforms.Compose(
            [
                transforms.Resize((480, 480)),
                transforms.ToTensor(),
                transforms.Normalize([0.568, 0.683, 0.597], [0.327, 0.302, 0.317])
            ]
        )

    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        model = osnet_x1_0(num_classes=1, loss='smoothL1Loss')
        load_pretrained_weights(model, 'last.pth')
        return model

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":".\/image\/0.jpg"}
        :return: 模型预测成功之后返回给系统样例 {"label":"3.71"}
        '''
        self.model.eval()
        cudnn.benchmark = True
        im = Image.open(image_path)
        im = self.transform(im)
        im = im.unsqueeze(0)
        im = im.cuda()
        out = self.model(im).cpu().detach().numpy()[0]
        # out = self.model(im)
        # print(out[0])
        out = out[0] * 5.0

        return {"label": str(out)}


if __name__ == '__main__':
    prediction = Prediction()
    result = prediction.predict('data/input/FacialBeautyPrediction/image/711.jpg')
    print(result)
