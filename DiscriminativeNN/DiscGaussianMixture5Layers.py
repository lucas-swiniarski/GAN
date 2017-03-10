import torch
import torch.nn as nn

class _netD(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD, self).__init__()
        self.main = nn.Sequential(

            nn.Linear(nc, ndf, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(ndf, ndf * 2, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(ndf * 2, ndf * 4, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(ndf * 4, ndf * 8, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(ndf * 8, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = nn.parallel.data_parallel(self.main, input, None)
        return output.view(-1, 1)
