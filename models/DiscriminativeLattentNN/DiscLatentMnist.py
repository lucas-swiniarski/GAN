import torch.nn as nn
import torch
import torch.nn.functional as F

class _netD(nn.Module):
    def __init__(self, nz, wasserstein):
        super(_netD, self).__init__()
        self.wasserstein = wasserstein
        self.fc1 = nn.Linear(nz, nz)
        self.bn1 = nn.BatchNorm1d(nz)
        self.fc2 = nn.Linear(nz, 1)

    def forward(self, input):
        input = self.bn1(self.fc1(input))
        F.leaky_relu(input, inplace=True)
        input = self.fc2(input)
        if self.wasserstein:
            input = F.sigmoid(input)
        return input
