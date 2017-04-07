import torch.nn as nn
import torch
import torch.nn.functional as F

class _netD(nn.Module):
    def __init__(self, ndf, nc, wasserstein, ac_gan, n_class, bias, dropout, bn_momentum, info_gan_latent, output_size=1):
        super(_netD, self).__init__()
        self.wasserstein = wasserstein
        self.ac_gan = ac_gan
        self.n_class = n_class
        self.fc_hidden_size = 250
        self.dropout = dropout
        self.bias = bias
        self.info_gan_latent = info_gan_latent

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(ndf * 2, momentum=bn_momentum)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=bias)
        self.bn3 = nn.BatchNorm2d(ndf * 4, momentum=bn_momentum)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=bias)
        self.bn4 = nn.BatchNorm2d(ndf * 8, momentum=bn_momentum)

        self.conv5 = nn.Conv2d(ndf * 8, self.output_size, 3, bias=bias)

        if info_gan_latent > 0:
            self.conv5_infogan = nn.Conv2d(ndf * 8, self.info_gan_latent, 3, bias=bias)
        if ac_gan:
            self.conv5_acgan = nn.Conv2d(ndf * 8, self.fc_hidden_size, 3, bias=bias)
            self.fc1 = nn.Linear(self.fc_hidden_size, self.n_class)

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

        input = self.bn4(self.conv4(input))

        if self.dropout:
            self.drop(input)
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        output = []
        output += [self.conv5(input).view(-1, self.output_size)]

        if not self.wasserstein:
            output[0] = F.sigmoid(output[0])

        if self.ac_gan:
            output += [self.fc1(F.relu(self.conv5_acgan(input).view(-1, self.fc_hidden_size)))]

        if self.info_gan_latent > 0:
            output += [F.sigmoid(self.conv5_infogan(input).view(-1, self.info_gan_latent))]

        return output
