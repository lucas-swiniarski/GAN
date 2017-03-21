import torch.nn as nn
import torch
import torch.nn.functional as F

class _netD(nn.Module):
    def __init__(self, ndf, nc, wasserstein, ac_gan, n_class, bias, dropout):
        super(_netD, self).__init__()
        self.wasserstein = wasserstein
        self.ac_gan = ac_gan
        self.n_class = n_class
        self.dropout = dropout

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=bias)
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=bias)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        if ac_gan:
            self.conv5 = nn.Conv2d(ndf * 8, 1 + self.n_class, 2, 1, 0, bias=bias)
        else:
            self.conv5 = nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=bias)
        if self.dropout:
            self.drop = nn.Dropout2d(inplace=True)

    def clamp(self, c, clamping_method):
        if clamping_method == 'clamp':
            self.conv1.weight.data.clamp_(-c, c)
            self.conv2.weight.data.clamp_(-c, c)
            self.conv3.weight.data.clamp_(-c, c)
            self.conv4.weight.data.clamp_(-c, c)
            self.conv5.weight.data[:,0].clamp_(-c, c)
        elif clamping_method == 'max_normalize':
            self.conv1.weight.data.div_(self.conv1.weight.data.abs().max()).mul_(c)
            self.conv2.weight.data.div_(self.conv2.weight.data.abs().max()).mul_(c)
            self.conv3.weight.data.div_(self.conv3.weight.data.abs().max()).mul_(c)
            self.conv4.weight.data.div_(self.conv4.weight.data.abs().max()).mul_(c)
            self.conv5.weight.data[:,0].div_(self.conv5.weight.data[:,0].abs().max()).mul_(c)
        elif clamping_method == 'normalize':
            self.conv1.weight.data.add_(-self.conv1.weight.data.mean()).div_(self.conv1.weight.data.std()).mul_(c)
            self.conv2.weight.data.add_(-self.conv2.weight.data.mean()).div_(self.conv2.weight.data.std()).mul_(c)
            self.conv3.weight.data.add_(-self.conv3.weight.data.mean()).div_(self.conv3.weight.data.std()).mul_(c)
            self.conv4.weight.data.add_(-self.conv4.weight.data.mean()).div_(self.conv4.weight.data.std()).mul_(c)
            self.conv5.weight.data[:,0].add_(-self.conv5.weight.data[:,0].mean()).div_(self.conv5.weight.data[:,0].std()).mul_(c)

    def forward(self, input):
        input = self.conv1(input)
        if self.dropout:
            self.drop(input)
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn2(self.conv2(input))
        if self.dropout:
            self.drop(input)
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn3(self.conv3(input))
        if self.dropout:
            self.drop(input)
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn4(self.conv4(input))
        if self.dropout:
            self.drop(input)
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.conv5(input)

        if not self.wasserstein:
            input[:,0] = F.sigmoid(input[:,0])

        if not self.ac_gan:
            return input.view(-1, 1)
        else:
            input = input.view(-1, 1 + self.n_class)
            return input[:,0], input[:,1:]
