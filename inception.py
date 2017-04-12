from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision
from torch.autograd import Variable

import sys
sys.path.append("..")
import utils
import functools

# For printing in real time on HPC
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

parser = argparse.ArgumentParser()
# Global parameters :
parser.add_argument('--dataset', type=str, default='mnist',help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../data', type=str, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--outf', default='../TrainedNetworks', help='folder to output images and model checkpoints')
parser.add_argument('--name', default='dcgan', help='Name used to save models')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--training-size', type=int, default=-1, help='How many examples of real data do we use, (default:-1 = Infinity)')

# Learning Parameters :
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--adam', action='store_true', help='Default RMSprop optimizer, wether to use Adam instead')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

# Generator Parameters :
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--bng-momentum', type=float, default=0.1, help='Momentum of BatchNorm Generator')
parser.add_argument('--model-g', type=str, default='base', help='Generator model : base | upsampling | residual')
parser.add_argument('--noise-unconnex', action='store_true', help='Train with an Unconnex Noise vector (No N(0,1))')

# Discriminator Parameters :
parser.add_argument('--noise', action='store_true', help='Add gaussian noise to real data')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--bnd-momentum', type=float, default=0.1, help='Momentum of BatchNorm Discriminator')
parser.add_argument('--dropout', action='store_true', help='Dropouts on discriminator')
parser.add_argument('--bias', action='store_true', help='Bias term on convolutions on discriminator')

# Gan type :
parser.add_argument('--n-critic', type=int, default=1,help='Times training the discriminator vs generator')
parser.add_argument('--clamp', action='store_true', help='Gan Discriminator')
parser.add_argument('--wasserstein', action='store_true', default=False, help='Training a Wasserstein Gan or a DC-GAN')
parser.add_argument('--c', type=float, default=0.01, help='Clamping parameter of the W-Gan')
parser.add_argument('--lbda', type=float, default=.3, help='Cycle loss factor')

#TODO : Implement grad-pen when backprop of gradients enabled
parser.add_argument('--grad-pen', action='store_true', default=False, help='Penalize Gradient')
parser.add_argument('--reg-factor', type=float, default=10, help='Improved WGAN training')

args = parser.parse_args()

args.manualSeed = 1 # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

print(args)

###
# Import Generative and Disriminative Networks, depending on the data-set ( image size).
###

sys.path.append("../models/GenerativeNN")

if args.dataset == 'cifar10':
    if args.model_g == 'base':
        import GenCifar10 as ModelGImage
    elif args.model_g == 'upsampling':
        import GenCifar10Upsampling as ModelGImage
    elif args.model_g == 'residual':
        import GenCifar10Residual as ModelGImage

    imageSize, nc = 32, 3
elif args.dataset == 'mnist':
    if args.model_g == 'base':
        import GenMnist as ModelGImage
    elif args.model_g == 'upsampling':
        import GenMnistUpsampling as ModelGImage
    elif args.model_g == 'residual':
        import GenMnistResidual as ModelGImage

    imageSize, nc = 28, 1
elif args.dataset == 'stl10':
    import GenStl10 as ModelGImage
    import DiscLatent as ModelDLatent
    import DiscStl10 as ModelMixed

    imageSize, nc = 64, 3

# TODO@Lucas : Import inception model, implement loss, print loss at the end. 
