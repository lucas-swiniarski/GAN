import torch.nn as nn
import torch
import torch.nn.functional as F

class _netD(nn.Module):
    def __init__(self, ndf, nc, wasserstein, ac_gan, n_class, bias, dropout, bn_momentum):
        super(_netD, self).__init__()
        self.wasserstein = wasserstein
        self.ac_gan = ac_gan
        self.n_class = n_class
        self.dropout = dropout
        self.bias = bias
        self.fc_hidden_size = 250

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(ndf * 2, momentum=bn_momentum)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=bias)
        self.bn3 = nn.BatchNorm2d(ndf * 4, momentum=bn_momentum)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=bias)
        self.bn4 = nn.BatchNorm2d(ndf * 8, momentum=bn_momentum)

        if ac_gan:
            self.conv5 = nn.Conv2d(ndf * 8, 1 + self.fc_hidden_size, 2, 1, 0, bias=bias)
            self.fc1 = nn.Linear(self.fc_hidden_size, self.n_class)
        else:
            self.conv5 = nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=bias)
        if self.dropout:
            self.drop = nn.Dropout2d(inplace=True)

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

        self.bn4.weight.data.clamp_(-0.01, 0.01)
        self.bn4.bias.data.clamp_(-0.01, 0.01)
        input = self.bn4(self.conv4(input))
        if self.dropout:
            self.drop(input)
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        self.conv5.weight[0].data.clamp_(-0.01, 0.01)
        input = self.conv5(input)

        if not self.wasserstein:
            input[:,0] = F.sigmoid(input[:,0])

        if not self.ac_gan:
            return input.view(-1, 1)
        else:
            input = input.view(-1, 1 + self.fc_hidden_size)
            return input[:,0], self.fc1(F.relu(input[:,1:]))
