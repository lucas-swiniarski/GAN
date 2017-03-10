import torch
import torch.nn as nn

class _netG(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z
            nn.Linear(nz, ngf * 8, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),

            nn.Linear(ngf * 8, ngf * 4, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),

            nn.Linear(ngf * 4, ngf * 2, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),

            nn.Linear(ngf * 2, ngf, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),

            nn.Linear(ngf, nc, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return nn.parallel.data_parallel(self.main, input, None)
