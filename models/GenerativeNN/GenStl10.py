import torch.nn as nn
import torch
import torch.nn.functional as F

class _netG(nn.Module):
    def __init__(self, n_input, ngf, nc, bn_momentum):
        super(_netG, self).__init__()
        self.convt1 = nn.ConvTranspose2d(n_input, ngf * 8, 8, 1, 0, bias=False)
        self.conv1_1 = nn.Conv2d(ngf * 8, ngf * 8, 3, padding=1)
        self.bn1_res_1 = nn.BatchNorm2d(ngf * 8, momentum=bn_momentum)
        self.conv1_2 = nn.Conv2d(ngf * 8, ngf * 8, 3, padding=1)
        self.bn1_res_2 = nn.BatchNorm2d(ngf * 8, momentum=bn_momentum)
        self.bn1 = nn.BatchNorm2d(ngf * 8, momentum=bn_momentum)

        self.convt2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.conv2_1 = nn.Conv2d(ngf * 4, ngf * 4, 3, padding=1)
        self.bn2_res_1 = nn.BatchNorm2d(ngf * 4, momentum=bn_momentum)
        self.conv2_2 = nn.Conv2d(ngf * 4, ngf * 4, 3, padding=1)
        self.bn2_res_2 = nn.BatchNorm2d(ngf * 4, momentum=bn_momentum)
        self.bn2 = nn.BatchNorm2d(ngf * 4, momentum=bn_momentum)

        self.convt3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False)
        self.conv3_1 = nn.Conv2d(ngf * 2, ngf * 2, 3, padding=1)
        self.bn3_res_1 = nn.BatchNorm2d(ngf * 2, momentum=bn_momentum)
        self.conv3_2 = nn.Conv2d(ngf * 2, ngf * 2, 3, padding=1)
        self.bn3_res_2 = nn.BatchNorm2d(ngf * 2, momentum=bn_momentum)
        self.bn3 = nn.BatchNorm2d(ngf * 2, momentum=bn_momentum)

        self.convt4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.conv4_1 = nn.Conv2d(ngf, ngf, 3, padding=1)
        self.bn4_res_1 = nn.BatchNorm2d(ngf, momentum=bn_momentum)
        self.conv4_2 = nn.Conv2d(ngf, ngf, 3, padding=1)
        self.bn4_res_2 = nn.BatchNorm2d(ngf, momentum=bn_momentum)
        self.bn4 = nn.BatchNorm2d(ngf, momentum=bn_momentum)

        self.convt5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        self.conv5_1 = nn.Conv2d(nc, nc, 3, padding=1)
        self.bn5_res_1 = nn.BatchNorm2d(nc, momentum=bn_momentum)
        self.conv5_2 = nn.Conv2d(nc, nc, 3, padding=1)
        self.bn5_res_2 = nn.BatchNorm2d(nc, momentum=bn_momentum)

    def forward(self, input):
        input = self.bn1(self.convt1(input))
        F.relu(input, inplace=True)
        input = input + F.relu(self.bn1_res_2(self.conv1_2(F.relu(self.bn1_res_1(self.conv1_1(input))))))

        input = self.bn2(self.convt2(input))
        F.relu(input, inplace=True)
        input = input + F.relu(self.bn2_res_2(self.conv2_2(F.relu(self.bn2_res_1(self.conv2_1(input))))))

        input = self.bn3(self.convt3(input))
        F.relu(input, inplace=True)
        input = input + F.relu(self.bn3_res_2(self.conv3_2(F.relu(self.bn3_res_1(self.conv3_1(input))))))

        input = self.bn4(self.convt4(input))
        F.relu(input, inplace=True)
        input = input + F.relu(self.bn4_res_2(self.conv4_2(F.relu(self.bn4_res_1(self.conv4_1(input))))))

        input = self.convt5(input)
        input = input + self.bn5_res_2(self.conv5_2(F.relu(self.bn5_res_1(self.conv5_1(input)))))
        input = F.tanh(input)
        return input
