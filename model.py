'''
Description: 网络模型
Author: wangdx
Date: 2021-06-12 18:07:33
LastEditTime: 2021-06-14 14:10:36
'''

"""
除了ResNet_*类之外，共包括两个类:
ResBlock2: 2层3*3卷积构成的block
ResBlock3: 3层卷积构成的block, 分别为1*1 3*3 1*1

ResNet中的_make_layer()函数将多个ResBlock组成layer
"""

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock2(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        
        stride_1 = 1 if in_c == out_c else 2
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride_1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        # 输入通道和输出通道不同时，跳跃连接需要下采样并增加通道数
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=2),    # 通过stride=2的1*1卷积实现下采样
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )
        else:
            self.shortcut = nn.Sequential()


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.shortcut(x) + out
        return out



class ResBlock3(nn.Module):
    def __init__(self, in_c, out_c, stride, downsample=False) -> None:
        """
        in_c: int 输入通道
        out_c: tuple(c1, c2, c3)
        stride: 第一个卷积的Stride
        """
        super().__init__()
        # 残差变换
        self.conv1 = nn.Conv2d(in_c, out_c[0], kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_c[0])
        self.conv2 = nn.Conv2d(out_c[0], out_c[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c[1])
        self.conv3 = nn.Conv2d(out_c[1], out_c[2], kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_c[2])
        # 恒等变换
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c[2], kernel_size=1, stride=stride), 
                nn.BatchNorm2d(out_c[2]),
                nn.ReLU()
            )
        else:
            self.shortcut = nn.Sequential()
            
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.shortcut(x) + out
        return out



class ResNet_34(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )      
        self.layer2 = self._make_layer(64, 64, 3)
        self.layer3 = self._make_layer(64, 128, 4)
        self.layer4 = self._make_layer(128, 256, 6)
        self.layer5 = self._make_layer(256, 512, 3)
        self.gap = nn.AdaptiveAvgPool2d((1,1)) # 全局平均池化
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)     # (batch, 512, 1, 1)
        x = self.fc(x.view(x.shape[0], -1))
        return x

    def _make_layer(self, in_c, out_c, num):
        layers = []
        # 先构建第一个block
        layers.append(ResBlock2(in_c, out_c))
        for _ in range(1, num):
            layers.append(ResBlock2(out_c, out_c))
        
        return nn.Sequential(*layers)



class ResNet_50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = self._make_layer(64, (64, 64, 256), stride=1, num=3)
        self.layer3 = self._make_layer(256, (128, 128, 512), stride=2, num=4)
        self.layer4 = self._make_layer(512, (256, 256, 1024), stride=2, num=6)
        self.layer5 = self._make_layer(1024, (512, 512, 2048), stride=2, num=3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x


    def _make_layer(self, in_c, out_c, stride, num):
        """
        in_c: int 输入通道
        out_c: tuple(c1, c2, c3)
        """
        layers = []
        # 先创建第一个block
        layers.append(ResBlock3(in_c, out_c, stride=stride, downsample=True))
        for _ in range(1, num):
            layers.append(ResBlock3(out_c[2], out_c, stride=1, downsample=False))
        return nn.Sequential(*layers)




def compute_loss(pred, target):
    """
    计算损失
    pred.shape = (batch, 2)
    target.shape = (batch, 1)
    """
    pred = torch.softmax(pred, dim=1)
    loss = F.cross_entropy(pred, target.view(target.shape[0]))
    return loss

def evaluation(pred, target):
    """
    评估预测正确或错误
    pred.shape = (batch, 2)
    target.shape = (batch, 1)

    return: 预测正确的数量
    """
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)    # (batch,)
    target = target.view(target.shape[0])   # (batch,)
    
    return torch.sum(torch.eq(pred, target)).item()

