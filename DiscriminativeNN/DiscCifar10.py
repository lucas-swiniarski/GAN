import torch.nn as nn
import torch
import torch.nn.functional as F

class _netD(nn.Module):
    def __init__(self, ndf, nc, wasserstein, dcgan, n_class):
        super(_netD, self).__init__()
        self.wasserstein = wasserstein
        self.dcgan = dcgan
        self.n_class = n_class

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False)
        if dcgan:
            self.conv5_classification = nn.Conv2d(ndf * 8, self.n_class, 2, 1, 0, bias=False)

    def forward(self, input):
        input = self.conv1(input)
        input = F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn2(self.conv2(input))
        input = F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn3(self.conv3(input))
        input = F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn4(self.conv4(input))
        input = F.leaky_relu(input, negative_slope=0.2, inplace=True)

        realfake = self.conv5(input)
        if not self.wasserstein:
            realfake = F.sigmoid(realfake)

        if not self.dcgan:
            return realfake.view(-1, 1)
        else:
            input = self.conv5_classification(input)
            return realfake.view(-1, 1), input.view(-1, self.n_class))
