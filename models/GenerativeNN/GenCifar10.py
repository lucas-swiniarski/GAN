import torch.nn as nn
import torch
import torch.nn.functional as F

class _netG(nn.Module):
    def __init__(self, n_input, ngf, nc, bn_momentum, discontinuity):
        super(_netG, self).__init__()
        self.discontinuity = discontinuity
        self.convt1 = nn.ConvTranspose2d(n_input, ngf * 8, 2, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8, momentum=bn_momentum)

        self.convt2 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 8, momentum=bn_momentum)

        self.convt3 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 8, momentum=bn_momentum)

        self.convt4 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf * 8, momentum=bn_momentum)

        self.convt5 = nn.ConvTranspose2d(ngf * 8, nc, 4, 2, 1, bias=False)

    def non_linearity(self, input):
        input = F.relu(input)
        if self.discontinuity:
            return input + torch.sign(input)
        return input

    def forward(self, input):
        input = self.bn1(self.convt1(input))
        F.relu(input, inplace=True)

        input = self.bn2(self.convt2(input))
        F.relu(input, inplace=True)

        input = self.bn3(self.convt3(input))
        F.relu(input, inplace=True)

        input = self.bn4(self.convt4(input))
        F.relu(input, inplace=True)

        input = self.convt5(input)
        input = F.tanh(input)
        return input
