import torch.nn as nn
import torch
import torch.nn.functional as F

class _netD(nn.Module):
    def __init__(self, ndf, nc, wasserstein, ac_gan, n_class, bias, dropout, bn_momentum, info_gan_latent, output_size=1):
        super(_netD, self).__init__()
        self.wasserstein = wasserstein
        self.ac_gan = ac_gan
        self.n_class = n_class
        self.dropout = dropout
        self.bias = bias
        self.fc_hidden_size = 250
        self.info_gan_latent = info_gan_latent
        self.output_size = output_size

        # 3 x 64 x 64
        self.conv1 = nn.Conv2d(nc, ndf, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(ndf, ndf, 3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.conv3 = nn.Conv2d(ndf, ndf, 3, padding=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(ndf)

        # ndf x 64 x 64
        self.conv4 = nn.Conv2d(ndf, ndf * 2, 3, stride=2, padding=1, bias=bias)
        self.bn4 = nn.BatchNorm2d(ndf * 2)
        self.conv5 = nn.Conv2d(ndf * 2, ndf * 2, 3, padding=1, bias=bias)
        self.bn5 = nn.BatchNorm2d(ndf * 2)
        self.conv6 = nn.Conv2d(ndf * 2, ndf * 2, 3, padding=1, bias=bias)
        self.bn6 = nn.BatchNorm2d(ndf * 2)
        # ndf * 2 x 32 x 32
        self.conv7 = nn.Conv2d(ndf * 2, ndf * 4, 3, stride=2,padding=1, bias=bias)
        self.bn7 = nn.BatchNorm2d(ndf * 4)
        self.conv8 = nn.Conv2d(ndf * 4, ndf * 4, 3, padding=1, bias=bias)
        self.bn8 = nn.BatchNorm2d(ndf * 4)
        self.conv9 = nn.Conv2d(ndf * 4, ndf * 4, 3, padding=1, bias=bias)
        self.bn9 = nn.BatchNorm2d(ndf * 4)
        # ndf * 4 x 16 x 16
        self.conv10 = nn.Conv2d(ndf * 4, ndf * 8, 3, padding=1, stride=2, bias=bias)
        self.bn10 = nn.BatchNorm2d(ndf * 8)
        self.conv11 = nn.Conv2d(ndf * 8, ndf * 8, 3, padding=1, bias=bias)
        self.bn11 = nn.BatchNorm2d(ndf * 8)
        self.conv12 = nn.Conv2d(ndf * 8, ndf * 8, 3, padding=1, bias=bias)
        self.bn12 = nn.BatchNorm2d(ndf * 8)
        # ndf * 8 x 8 x 8
        self.conv13 = nn.Conv2d(ndf * 8, self.output_size, 8)

    def forward(self, input):
        input = self.conv1(input)
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn2(self.conv2(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn3(self.conv3(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn4(self.conv4(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn5(self.conv5(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn6(self.conv6(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn7(self.conv7(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn8(self.conv8(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn9(self.conv9(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn10(self.conv10(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)
        print(input.size())
        input = self.bn11(self.conv11(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        input = self.bn12(self.conv12(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        output = []
        output += [self.conv13(input).view(-1, self.output_size)]

        if not self.wasserstein:
            output[0] = F.sigmoid(output[0])
        return output
