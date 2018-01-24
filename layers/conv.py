import torch
from torch import nn


class MultiDilationConv2d(nn.Module):
    def __init__(self, _in: int, _out: int, dilations: tuple, kernel_size=3, stride=1, bn=False, activation=nn.ReLU):
        super().__init__()
        assert len(dilations) != 0
        self.convs = nn.ModuleList([self.__padded_conv__(_in, _out, kernel_size, stride, d, bn, activation) for d in dilations])
        self.final = nn.Conv2d(len(dilations) * _out, _out, kernel_size=1)

    @staticmethod
    def __padded_conv__(_in: int, _out: int, kernel_size:int, stride=1, dilation=1, bn=False, activation=None):
        padding = (kernel_size // 2) * dilation
        layers = []
        conv = nn.Conv2d(_in, _out, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=False)
        layers.append(conv)
        if bn:
            layers.append(nn.BatchNorm2d(_out))
        if activation is not None:
            layers.append(activation())
        return layers[0] if len(layers) == 1 else nn.Sequential(*layers)

    def forward(self, x):
        feats = []
        for _, mod in enumerate(self.convs):
            feats.append(mod(x))
        return self.final(torch.cat(feats, dim=1))
