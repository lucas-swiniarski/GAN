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

### Wasserstein plot discriminator values

def plot_discriminator(X,Y,Z):
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
