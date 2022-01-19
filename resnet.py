import numpy as np
import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)  


from collections import OrderedDict

class ResNetGate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.blocks = nn.Sequential(OrderedDict({
            'conv': Conv2dAuto(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False)
            ,'bn': nn.BatchNorm2d(out_channels)
            ,'activation': nn.ReLU()
            ,'mp': nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            }))
    
    def forward(self, x):
        return self.blocks(x)


from collections import OrderedDict

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv=conv3x3, activation=nn.ReLU):
        super().__init__()
        self.in_channels, self.out_channels, self.conv = in_channels, out_channels, conv

        stride = 2 if in_channels != out_channels else 1

        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1,
                      stride=2, bias=False),
            nn.BatchNorm2d(self.out_channels)
        ) if self.apply_shortcut else None

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, stride=stride)
            ,activation()
            ,conv_bn(self.out_channels, self.out_channels, self.conv)
        )

    @property
    def apply_shortcut(self):
        return self.in_channels != self.out_channels
    
    def forward(self, x):
        residual = x
        if self.apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, activation=nn.ReLU, *args, **kwargs):
        super().__init__()

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, activation=activation, *args, **kwargs),
            *[block(out_channels, out_channels, activation=activation, *args, **kwargs) for n in range(n-1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, layer_sizes=[2,2,2,2], block_sizes=[64, 128, 256, 512]
                ,activation=nn.ReLU, block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.gate = ResNetGate(in_channels, block_sizes[0], kernel_size=7, stride=2)

        self.in_out_block_sizes = list(zip(block_sizes, block_sizes[1:]))

        self.encoder = nn.ModuleList([
            ResNetLayer(block_sizes[0], block_sizes[0], block=block, n=layer_sizes[0], activation=activation)
            ,*[ResNetLayer(in_channels, out_channels, block=block, n=depth, activation=activation) 
                for (in_channels, out_channels), depth in zip(self.in_out_block_sizes, layer_sizes[1:])]]
        )

    def forward(self, x):
        x = self.gate(x)
        for layer in self.encoder:
            x = layer(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, in_features, n_classes, *args, **kwargs):
        super().__init__()

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()

        self.encoder = ResNetEncoder(in_channels=in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(self.encoder.encoder[-1].blocks[-1].out_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, layer_sizes=[3, 4, 6, 3])

def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, layer_sizes=[2, 2, 2, 2])
