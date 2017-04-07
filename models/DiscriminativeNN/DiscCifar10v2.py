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
        self.fc_hidden_size = ndf * 8

        # 3 x 32 x 32
        self.conv1 = nn.Conv2d(nc, ndf, 3, stride=2, padding=1, bias=bias)
        # ndf x 16 x 16
        self.conv1_1 = nn.Conv2d(ndf, ndf, 3, padding=1, bias=bias)
        self.bn1_1 = nn.BatchNorm2d(ndf, momentum=bn_momentum)
        self.conv1_2 = nn.Conv2d(ndf, ndf, 3, padding=1, bias=bias)
        self.bn1_2 = nn.BatchNorm2d(ndf, momentum=bn_momentum)
        self.bn1_3 = nn.BatchNorm2d(ndf, momentum=bn_momentum)

        # ndf x 16 x 16
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 3, 2, padding=1, bias=bias)
        # ndf * 2 x 8 x 8
        self.bn2 = nn.BatchNorm2d(ndf * 2, momentum=bn_momentum)

        self.conv2_1 = nn.Conv2d(ndf * 2, ndf * 2, 3, padding=1, bias=bias)
        self.bn2_1 = nn.BatchNorm2d(ndf * 2, momentum=bn_momentum)
        self.conv2_2 = nn.Conv2d(ndf * 2, ndf * 2, 3, padding=1, bias=bias)
        self.bn2_2 = nn.BatchNorm2d(ndf * 2, momentum=bn_momentum)
        self.bn2_3 = nn.BatchNorm2d(ndf * 2, momentum=bn_momentum)

        # ndf * 2 x 8 x 8
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=bias)
        # ndf * 4 x 4 x 4
        self.bn3 = nn.BatchNorm2d(ndf * 4, momentum=bn_momentum)

        self.conv3_1 = nn.Conv2d(ndf * 4, ndf * 4, 3, padding=1, bias=bias)
        self.bn3_1 = nn.BatchNorm2d(ndf * 4, momentum=bn_momentum)
        self.conv3_2 = nn.Conv2d(ndf * 4, ndf * 4, 3, padding=1, bias=bias)
        self.bn3_2 = nn.BatchNorm2d(ndf * 4, momentum=bn_momentum)
        self.bn3_3 = nn.BatchNorm2d(ndf * 4, momentum=bn_momentum)

        # ndf * 4 x 4 x 4
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 3, stride=2, padding=1, bias=bias)
        # ndf * 8 x 2 x 2
        self.bn4 = nn.BatchNorm2d(ndf * 8, momentum=bn_momentum)

        self.conv4_1 = nn.Conv2d(ndf * 8, ndf * 8, 3, padding=1, bias=bias)
        self.bn4_1 = nn.BatchNorm2d(ndf * 8, momentum=bn_momentum)
        self.conv4_2 = nn.Conv2d(ndf * 8, ndf * 8, 3, padding=1, bias=bias)
        self.bn4_2 = nn.BatchNorm2d(ndf * 8, momentum=bn_momentum)
        self.bn4_3 = nn.BatchNorm2d(ndf * 8, momentum=bn_momentum)

        if ac_gan:
            self.conv5 = nn.Conv2d(ndf * 8, 1 + self.fc_hidden_size, 2, 1, 0, bias=bias)
            self.fc1 = nn.Linear(self.fc_hidden_size, self.fc_hidden_size)
            self.bn6 = nn.BatchNorm1d(self.fc_hidden_size)
            self.fc2 = nn.Linear(self.fc_hidden_size, self.n_class)
        else:
            self.conv5 = nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=bias)

    def forward(self, input):
        input = self.conv1(input)
        F.leaky_relu(input, negative_slope=0.2, inplace=True)
        input = input + F.leaky_relu(self.bn1_2(self.conv1_2(F.leaky_relu(self.bn1_1(self.conv1_1(input)), negative_slope=0.2))), negative_slope=0.2)
        input = self.bn2(self.conv2(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)
        input = input + F.leaky_relu(self.bn2_2(self.conv2_2(F.leaky_relu(self.bn2_1(self.conv2_1(input)), negative_slope=0.2))), negative_slope=0.2)
        input = self.bn3(self.conv3(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)
        input = input + F.leaky_relu(self.bn3_2(self.conv3_2(F.leaky_relu(self.bn3_1(self.conv3_1(input)), negative_slope=0.2))), negative_slope=0.2)
        input = self.bn4(self.conv4(input))
        F.leaky_relu(input, negative_slope=0.2, inplace=True)

        self.bn4_3.weight.data.clamp_(-0.01,0.01)
        self.bn4_3.bias.data.clamp_(-0.01,0.01)
        
        input = self.bn4_3(input + F.leaky_relu(self.bn4_2(self.conv4_2(F.leaky_relu(self.bn4_1(self.conv4_1(input)), negative_slope=0.2))), negative_slope=0.2))
        self.conv5.weight[0].data.clamp_(-0.01,0.01)
        input = self.conv5(input)
        if not self.wasserstein:
            input[:,0] = F.sigmoid(input[:,0])

        if not self.ac_gan:
            return input.view(-1, 1)
        else:
            input = input.view(-1, 1 + self.fc_hidden_size)
            return input[:,0], self.fc2(self.bn6(self.fc1(F.relu(input[:,1:]))))
