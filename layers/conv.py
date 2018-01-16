import torch
from torch import nn


class ParallelDilated(nn.Module):
    def __init__(self, _in: int, _out: int, dilations: tuple, kernel_size=3, bn=False):
        super(ParallelDilated, self).__init__()
        assert len(dilations) != 0
        self.convs = nn.ModuleList([self.__make_conv_block__(_in, kernel_size, d, bn) for d in dilations])
        self.final = nn.Conv2d(len(dilations) * _in, _out, kernel_size=1, bias=False)

    @staticmethod
    def __make_conv_block__(_in, kernel_size, dilation, bn=False):
        padding = (kernel_size // 2) * dilation
        mod = [nn.Conv2d(_in, _in,
                         kernel_size=kernel_size, stride=1, dilation=dilation, padding=padding, bias=False)]
        if bn:
            mod.append(nn.BatchNorm2d(_in))
        mod.append(nn.ReLU())
        return nn.Sequential(*mod)

    def forward(self, x):
        feats = []
        for _, mod in enumerate(self.convs):
            feats.append(mod(x))
        return self.final(torch.cat(feats, dim=1))
