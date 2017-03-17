import torch.nn as nn
import torch
import torch.nn.functional as F

class _netD(nn.Module):
    def __init__(self, ndf, nc, wasserstein, ac_gan, n_class):
        super(_netD, self).__init__()
        self.wasserstein = wasserstein
        self.ac_gan = ac_gan
        self.n_class = n_class

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        if ac_gan:
            self.conv5 = nn.Conv2d(ndf * 8, 1 + self.n_class, 3, bias=False)
        else:
            self.conv5 = nn.Conv2d(ndf * 8, 1, 3, bias=False)

    def forward(self, input):
        input = self.conv1(input)
        F.leaky_relu(input, negative_slope=0.2, inplace=True)
        
        input = self.bn2(self.conv2(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn3(self.conv3(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn4(self.conv4(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.conv5(input)

        if not self.wasserstein:
            input[:,0] = F.sigmoid(input[:,0])

        if not self.ac_gan:
            return input.view(-1, 1)
        else:
            input = input.view(-1, 1 + self.n_class)
            return input[:,0], input[:,1:]
