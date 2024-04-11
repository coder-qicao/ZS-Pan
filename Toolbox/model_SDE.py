import torch
from torch import nn


class Residual_Block(nn.Module):
    def __init__(self,channels):
        super(Residual_Block, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,channels,3,1,1)
        )

    def forward(self,x):
        x=self.conv(x)+x
        return x


class Net_ms2pan(nn.Module):
    def __init__(self):
        super(Net_ms2pan, self).__init__()
        self.ups = nn.UpsamplingBilinear2d(scale_factor=4)
        self.net = nn.Sequential(nn.Linear(8, 32),
                                 nn.Sigmoid(),
                                 nn.Linear(32, 32),
                                 nn.Sigmoid(),
                                 nn.Linear(32, 1),)

    def forward(self, ms):
        out = self.net(ms.permute(0, 2, 3, 1))
        return out.permute(0, 3, 1, 2)

# class Net_ms2pan(nn.Module):
#     def __init__(self):
#         super(Net_ms2pan, self).__init__()
#         self.ups = nn.UpsamplingBilinear2d(scale_factor=4)
#         self.net = nn.Sequential(nn.Conv2d(8,32,3,1,1),
#                                nn.ReLU(inplace=True),
#                                Residual_Block(32),
#                                Residual_Block(32),
#                                Residual_Block(32),
#                                nn.Conv2d(32,1,3,1,1))
#
#     def forward(self, ms):
#         out = self.net(ms)
#         return out
