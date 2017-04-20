import torch.nn as nn
import torch
import torch.nn.functional as F

class _netD(nn.Module):
    def __init__(self, ndf, nz, wasserstein, bias, bn_momentum):
        super(_netD, self).__init__()
        self.bias = bias
        self.bn_momentum = bn_momentum

        self.wasserstein = wasserstein
        self.conv1 = nn.Conv2d(nz, ndf * 16, 1, bias=self.bias)
        self.drop1 = nn.Dropout(p=0)
        self.conv2 = nn.Conv2d(ndf * 16, ndf * 16, 1, bias=self.bias)
        self.bn2 = nn.BatchNorm2d(ndf * 16, momentum=bn_momentum)
        self.drop2 = nn.Dropout(p=0)
        self.conv3 = nn.Conv2d(ndf * 16, ndf * 16, 1, bias=self.bias)
        self.bn3 = nn.BatchNorm2d(ndf * 16, momentum=bn_momentum)
        self.drop3 = nn.Dropout(p=0)
        self.conv4 = nn.Conv2d(ndf * 16, ndf * 16, 1, bias=self.bias)
        self.bn4 = nn.BatchNorm2d(ndf * 16, momentum=bn_momentum)
        self.drop4 = nn.Dropout(p=0)
        self.conv5 = nn.Conv2d(ndf * 16, 1, 1, bias=self.bias)
        self.drop5 = nn.Dropout(p=0)

    def forward(self, input):
        input = self.drop1(self.conv1(input))
        F.leaky_relu(input, inplace=True, negative_slope=0.2)
        input = self.drop2(self.bn2(self.conv2(input)))
        F.leaky_relu(input, inplace=True, negative_slope=0.2)
        input = self.drop3(self.bn3(self.conv3(input)))
        F.leaky_relu(input, inplace=True, negative_slope=0.2)
        input = self.drop4(self.bn4(self.conv4(input)))
        F.leaky_relu(input, inplace=True, negative_slope=0.2)
        input = self.drop5(self.conv5(input))
        if not self.wasserstein:
            input = F.sigmoid(input)
        output = []

        # Possibility in the future to have multiple heads in this discriminator.
        output += [input]
        return output
