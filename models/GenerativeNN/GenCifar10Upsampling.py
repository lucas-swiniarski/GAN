import torch.nn as nn
import torch
import torch.nn.functional as F

class _netG(nn.Module):
    def __init__(self, n_input, ngf, nc, bn_momentum):
        super(_netG, self).__init__()
        self.convt1 = nn.ConvTranspose2d(n_input, ngf * 8, 2, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.conv2 = nn.Conv2d(ngf * 8, ngf * 8, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 8)

        self.upsample3 = nn.UpsamplingNearest2d(size=(4,4))
        self.conv3 = nn.Conv2d(ngf * 8, ngf * 8, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 8)
        self.conv4 = nn.Conv2d(ngf * 8, ngf * 4, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf * 4)

        self.upsample5 = nn.UpsamplingNearest2d(size=(8,8))
        self.conv5 = nn.Conv2d(ngf * 4, ngf * 4, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf * 4)
        self.conv6 = nn.Conv2d(ngf * 4, ngf * 2, 3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(ngf * 2)

        self.upsample7 = nn.UpsamplingNearest2d(size=(16,16))
        self.conv7 = nn.Conv2d(ngf * 2, ngf * 2, 3,  padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(ngf * 2)
        self.conv8 = nn.Conv2d(ngf * 2, ngf, 3,  padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(ngf)

        self.upsample9 = nn.UpsamplingNearest2d(size=(32,32))
        self.conv9 = nn.Conv2d(ngf, ngf, 3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(ngf)
        self.conv10 = nn.Conv2d(ngf, nc, 3, padding=1, bias=False)

    def forward(self, input):
        input = self.bn1(self.convt1(input))
        F.relu(input, inplace=True)
        input = self.bn2(self.conv2(input))
        F.relu(input, inplace=True)
        input = self.bn3(self.conv3(self.upsample3(input)))
        F.relu(input, inplace=True)
        input = self.bn4(self.conv4(input))
        F.relu(input, inplace=True)
        input = self.bn5(self.conv5(self.upsample5(input)))
        F.relu(input, inplace=True)
        input = self.bn6(self.conv6(input))
        F.relu(input, inplace=True)
        input = self.bn7(self.conv7(self.upsample7(input)))
        F.relu(input, inplace=True)
        input = self.bn8(self.conv8(input))
        F.relu(input, inplace=True)
        input = self.bn9(self.conv9(self.upsample9(input)))
        F.relu(input, inplace=True)
        input = self.conv10(input)
        input = F.tanh(input)
        return input
