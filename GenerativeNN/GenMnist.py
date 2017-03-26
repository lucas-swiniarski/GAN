import torch.nn as nn
import torch
import torch.nn.functional as F

class _netG(nn.Module):
    def __init__(self, n_input, ngf, nc, bn_momentum):
        super(_netG, self).__init__()
        self.convt1 = nn.ConvTranspose2d(n_input, ngf * 8, 2, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8, momentum=bn_momentum)

        self.convt2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4, momentum=bn_momentum)

        self.convt3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2, momentum=bn_momentum)

        self.convt4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf, momentum=bn_momentum)

        self.convt5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

    def forward(self, input):
        print('Forward ...')
        print(input.size())
        input = self.bn1(self.convt1(input))
        print(input.size())
        F.relu(input, inplace=True)

        input = self.bn2(self.convt2(input))
        print(input.size())
        F.relu(input, inplace=True)

        input = self.bn3(self.convt3(input))
        print(input.size())
        F.relu(input, inplace=True)

        input = self.bn4(self.convt4(input))
        print(input.size())
        F.relu(input, inplace=True)

        input = self.convt5(input)
        print(input.size())
        print('Stop ...')
        input = F.tanh(input)
        return input
