import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision.utils

class Residual_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super(Residual_Block, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            )
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False),
                                          nn.BatchNorm2d(out_c),)
        else:
            self.shortcut = nn.Sequential()

    def forward (self,x):
        residual = self.shortcut(x)
        x = self.feature(x)
        x = nn.functional.leaky_relu(residual + x, inplace=True)
        return x

class resnet(nn.Module):
    def __init__(self, widen_factor=4):
        super(resnet, self).__init__()
        self.widen_factor = widen_factor
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias = False)
        self.res_block1 = Residual_Block(64, 64)
        self.res_block2 = Residual_Block(64, 64)
        self.res_block3 = Residual_Block(64, 128)
        self.res_block4 = Residual_Block(128, 128)
        self.res_block5 = Residual_Block(128, 256)
        self.res_block6 = Residual_Block(256, 256)
        self.fc_1 = nn.Linear(256*3*3, 100)

    def forward(self, x):
        x = self.conv_1(x)                          #(B, 16, H, W)
        x = self.res_block1(x)
        x = F.max_pool2d(self.res_block2(x), 2)     #(B, 64, H/2, W/2)
        x = self.res_block3(x)
        x = F.max_pool2d(self.res_block4(x), 4)     #(B, 128, H/8, W/8)
        x = self.res_block5(x)
        latent = F.avg_pool2d(self.res_block6(x), 4) #(B, 256, H/32, W/32)
        x = latent.view(latent.size(0), -1)
        x = self.fc_1(x)
        return x, latent