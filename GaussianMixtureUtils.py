from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Gaussian Mixture Hyperparameters

mu_1, sigma_1 = (0.5, 0.5), (0.01, 0.01)
mu_2, sigma_2 = (-0.5, -0.5), (0.01, 0.01)

### Get n samples from the multi gaussian defined above.
### Return torch tensor.

def sample(n=1):
    rand_int = np.random.randint(2, size=n)
    rand_int = np.transpose(np.tile(rand_int, (2, 1)))
    return torch.from_numpy(rand_int * np.random.normal(mu_1, sigma_1, size=(n, 2)) + (1 - rand_int) * np.random.normal(mu_2, sigma_2, size=(n, 2)))

### Old plot functions
### Plot histogram of a an array (n samples x 2)

def plot_samples(SamplesToPlot):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Samples = SamplesToPlot.numpy().transpose()
    x, y = Samples[0], Samples[1]
    hist, xedges, yedges = np.histogram2d(x, y, bins=32, range=[[-1, 1], [-1, 1]])

    # Construct arrays for the anchor positions of the 16 bars.
    # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
    # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
    # with indexing='ij'.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    plt.show()

### Plot a heat map from an array of 2d samples.

def plot_heat_map(SamplesToPlot):
    Samples = SamplesToPlot.numpy().transpose()
    x, y = Samples[0], Samples[1]
    hist, xedges, yedges = np.histogram2d(x, y, bins=32, range=[[-1, 1], [-1, 1]])
    zmin, zmax = 0, 1
    CS = plt.contourf(xedges[:-1], yedges[:-1], hist / hist.max(), 15, cmap=plt.cm.rainbow,
                      vmax=zmax, vmin=zmin)
    plt.colorbar()
    plt.show()

def plot_duo_heat_map(SamplesToPlot, netD):
    # Two subplots, unpack the axes array immediately
    plt.close('all')
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,4))

    ### PLot Heatmap Samples ...

    Samples = SamplesToPlot.numpy().transpose()
    x, y = Samples[0], Samples[1]
    hist, xedges, yedges = np.histogram2d(x, y, bins=32, range=[[-1, 1], [-1, 1]])
    zmin, zmax = 0, 1
    CS = ax1.contourf(xedges[:-1], yedges[:-1], hist / hist.max(), 15, cmap=plt.cm.rainbow,
                      vmax=zmax, vmin=zmin)

    ax1.set_title("Generator distribution")
    f.colorbar(CS, ax=ax1)

    ### Plot Discriminator

    delta = 1.0 / 50
    x = np.arange(-1.0, 1.0, delta)
    y = np.arange(-1.0, 1.0, delta)
    X, Y = np.meshgrid(x, y)
    z = np.zeros((X.shape[0] * X.shape[1], 2))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z[i * X.shape[0] + j] = np.append(X[i][j], Y[i][j])

    input = torch.FloatTensor(X.shape[0] * X.shape[1], 2)
    real_cpu = torch.FloatTensor(z)
    batch_size = real_cpu.size(0)

    input = Variable(input)
    input.data.resize_(real_cpu.size()).copy_(real_cpu)
    output_real = netD(input).data.numpy()

    CS2 = ax2.pcolormesh(X, Y, output_real.reshape(X.shape))
    f.colorbar(CS2, ax=ax2)
    ax2.set_title("Discriminator")

    ###

    plt.plot()



### Wasserstein plot discriminator values

def plot_discriminator(netD):
    delta = 1.0 / 50
    x = np.arange(-1.0, 1.0, delta)
    y = np.arange(-1.0, 1.0, delta)
    X, Y = np.meshgrid(x, y)
    z = np.zeros((X.shape[0] * X.shape[1], 2))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z[i * X.shape[0] + j] = np.append(X[i][j], Y[i][j])

    input = torch.FloatTensor(X.shape[0] * X.shape[1], 2)
    real_cpu = torch.FloatTensor(z)
    batch_size = real_cpu.size(0)

    input = Variable(input)
    input.data.resize_(real_cpu.size()).copy_(real_cpu)
    output_real = netD(input).data.numpy()

    CS = plt.pcolormesh(X, Y, output_real.reshape(X.shape))
    plt.colorbar()
    plt.show()
