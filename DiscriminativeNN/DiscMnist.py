import torch.nn as nn
import torch
import torch.nn.functional as F

class _netD(nn.Module):
    def __init__(self, ndf, nc, wasserstein, ac_gan, n_class, bias, dropout):
        super(_netD, self).__init__()
        self.wasserstein = wasserstein
        self.ac_gan = ac_gan
        self.n_class = n_class
        self.fc_hidden_size = 250
        self.dropout = dropout

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=bias)
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=bias)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        if ac_gan:
            self.conv5 = nn.Conv2d(ndf * 8, 1 + self.fc_hidden_size, 3, bias=bias)
            self.fc1 = nn.Linear(self.fc_hidden_size, self.n_class)
        else:
            self.conv5 = nn.Conv2d(ndf * 8, 1, 3, bias=bias)
        if self.dropout:
            self.drop = nn.Dropout2d(inplace=True)

    def clamp(self, c, clamping_method):
        if clamping_method == 'clamp':
            self.conv1.weight.data.clamp_(-c, c)

            self.conv2.weight.data.clamp_(-c, c)
            self.bn2.weight.data.clamp_(-c, c)

            self.conv3.weight.data.clamp_(-c, c)
            self.bn3.weight.data.clamp_(-c, c)

            self.conv4.weight.data.clamp_(-c, c)
            self.bn4.weight.data.clamp_(-c, c)

            self.conv5.weight.data[0].clamp_(-c, c)

            if self.bias:
                self.conv1.bias.data.clamp_(-c,c)
                self.conv2.bias.data.clamp_(-c,c)
                self.conv3.bias.data.clamp_(-c,c)
                self.conv4.bias.data.clamp_(-c,c)
                self.conv5.bias[0].data.clamp_(-c,c)
        elif clamping_method == 'max_normalize':
            self.conv1.weight.data.div_(self.conv1.weight.data.abs().max()).mul_(c)

            self.conv2.weight.data.div_(self.conv2.weight.data.abs().max()).mul_(c)
            self.bn2.weight.data.div_(self.bn2.weight.data.abs().max()).mul_(c)

            self.conv3.weight.data.div_(self.conv3.weight.data.abs().max()).mul_(c)
            self.bn3.weight.data.div_(self.bn3.weight.data.abs().max()).mul_(c)

            self.conv4.weight.data.div_(self.conv4.weight.data.abs().max()).mul_(c)
            self.bn4.weight.data.div_(self.bn4.weight.data.abs().max()).mul_(c)

            self.conv5.weight.data[0].div_(self.conv5.weight.data[0].abs().max()).mul_(c)

            if self.bias:
                self.conv1.bias.data.div_(self.conv1.bias.data.abs().max()).mul_(c)
                self.conv2.bias.data.div_(self.conv2.bias.data.abs().max()).mul_(c)
                self.conv3.bias.data.div_(self.conv3.bias.data.abs().max()).mul_(c)
                self.conv4.bias.data.div_(self.conv4.bias.data.abs().max()).mul_(c)
                self.conv5.bias[0].data.clamp_(-c,c)
        elif clamping_method == 'normalize':
            self.conv1.weight.data.add_(-self.conv1.weight.data.mean()).div_(self.conv1.weight.data.std()).mul_(c)

            self.conv2.weight.data.add_(-self.conv2.weight.data.mean()).div_(self.conv2.weight.data.std()).mul_(c)
            self.bn2.weight.data.add_(-self.bn2.weight.data.mean()).div_(self.bn2.weight.data.std()).mul_(c)

            self.conv3.weight.data.add_(-self.conv3.weight.data.mean()).div_(self.conv3.weight.data.std()).mul_(c)
            self.bn3.weight.data.add_(-self.bn3.weight.data.mean()).div_(self.bn3.weight.data.std()).mul_(c)

            self.conv4.weight.data.add_(-self.conv4.weight.data.mean()).div_(self.conv4.weight.data.std()).mul_(c)
            self.bn4.weight.data.add_(-self.bn4.weight.data.mean()).div_(self.bn4.weight.data.std()).mul_(c)

            self.conv5.weight.data[0].add_(-self.conv5.weight.data[0].mean()).div_(self.conv5.weight.data[0].std()).mul_(c)

            if self.bias:
                self.conv1.bias.data.add_(-self.conv1.bias.data.mean()).div_(self.conv1.bias.data.std()).mul_(c)
                self.conv2.bias.data.add_(-self.conv2.bias.data.mean()).div_(self.conv2.bias.data.std()).mul_(c)
                self.conv3.bias.data.add_(-self.conv3.bias.data.mean()).div_(self.conv3.bias.data.std()).mul_(c)
                self.conv4.bias.data.add_(-self.conv4.bias.data.mean()).div_(self.conv4.bias.data.std()).mul_(c)
                self.conv5.bias[0].data.clamp_(-c, c)

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
            input = input.view(-1, 1 + self.fc_hidden_size)
            return input[:,0], self.fc1(F.relu(input[:,1:]))
