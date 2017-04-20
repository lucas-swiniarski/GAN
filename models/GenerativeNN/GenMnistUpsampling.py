import torch.nn as nn
import torch
import torch.nn.functional as F

class _netG(nn.Module):
    def __init__(self, n_input, ngf, nc, bn_momentum, discontinuity):
        super(_netG, self).__init__()
        self.discontinuity = discontinuity
        self.convt1 = nn.ConvTranspose2d(n_input, ngf * 8, 2, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8, momentum=bn_momentum)

        self.upsample2 = nn.UpsamplingBilinear2d(size=(4,4))
        self.conv2 = nn.Conv2d( ngf * 8,  ngf * 4, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4, momentum=bn_momentum)

        self.upsample3 = nn.UpsamplingBilinear2d(size=(7,7))
        self.conv3 = nn.Conv2d(ngf * 4, ngf * 2, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2, momentum=bn_momentum)

        self.upsample4 = nn.UpsamplingBilinear2d(size=(14,14))
        self.conv4 = nn.Conv2d(ngf * 2, ngf, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf, momentum=bn_momentum)

        self.upsample5 = nn.UpsamplingBilinear2d(size=(28,28))
        self.conv5 = nn.Conv2d(ngf, nc, 3, padding=1, bias=False)

    def non_linearity(self, input):
        input = F.relu(input)
        if self.discontinuity:
            return input + torch.sign(input)
        return input

    def forward(self, input):
        input = self.bn1(self.convt1(input))
        F.relu(input, inplace=True)

        input = self.bn2(self.conv2(self.upsample2(input)))
        F.relu(input, inplace=True)

        input = self.bn3(self.conv3(self.upsample3(input)))
        F.relu(input, inplace=True)

        input = self.bn4(self.conv4(self.upsample4(input)))
        F.relu(input, inplace=True)

        input = self.conv5(self.upsample5(input))
        input = F.tanh(input)
        return input
