"""
Mode dropping metric for Generative Adversarial Networks :
This python file takes a trained Generator and outputs the number of unique nearest neighbors (based on some threshold) in Dataset generated by the generator.
"""

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import faiss
import sys
import utils
import numpy as np
import matplotlib.pyplot as plt

# For printing in real time on cluster
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

parser = argparse.ArgumentParser()
# Global parameters :
parser.add_argument('--dataset', type=str, default='mnist',help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='data', type=str, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')

# Generator parameters :
parser.add_argument('--netG', type=str, required=True, help='Trained Generator Network Path')
parser.add_argument('--model-g', type=str, default='base', help='Generator model')
parser.add_argument('--nz', type=int, default=100, help='Latent size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--bng-momentum', type=float, default=0.1, help='Momentum of BatchNorm Generator')
parser.add_argument('--noise-unconnex', action='store_true', help='Train with an Unconnex Noise vector (No N(0,1))')

# k-NN parameters :
parser.add_argument('--k', type=int, default=4, help='k-NN')
parser.add_argument('--n-gen', type=int, default=2, help='Number of Elements generated.')
parser.add_argument('--threshold', type=float, default=.05, help='threshold distance after where no neighbors')

args = parser.parse_args()

args.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

print(args)

sys.path.append("models/DiscriminativeNN")
sys.path.append("models/GenerativeNN")

if args.dataset == 'cifar10':
    if args.model_g == 'base':
        import GenCifar10 as ModelG
    elif args.model_g == 'upsampling':
        import GenCifar10Upsampling as ModelG
    elif args.model_g == 'residual':
        import GenCifar10Residual as ModelG
    imageSize, nc = 32, 3
elif args.dataset == 'mnist':
    if args.model_g == 'base':
        import GenMnist as ModelGImage
    elif args.model_g == 'upsampling':
        import GenMnistUpsampling as ModelGImage
    elif args.model_g == 'residual':
        import GenMnistResidual as ModelGImage
    imageSize, nc = 28, 1

trainloader, testloader, n_class = utils.load_dataset(args.dataset, args.dataroot, args.batchSize, imageSize, args.workers, -1)

assert trainloader, testloader

netGImage = ModelGImage._netG(args.nz, args.ngf, nc, args.bng_momentum)
netGImage.load_state_dict(torch.load(args.netG))

# Read the data-set :
dim = nc * imageSize * imageSize
index = faiss.IndexFlatL2(dim)
print(index.is_trained)

for i, (data, _) in enumerate(trainloader, 0):
    data.resize_(data.size(0), dim)
    index.add(data.numpy())
print('k-NN index size : {}'.format(index.ntotal))

latent = torch.FloatTensor(args.batchSize, args.nz, 1, 1)
if args.cuda:
    latent = latent.cuda()
    netGImage.cuda()
latent = Variable(latent)

found_img = {}
X = []

for i in range(1, args.n_gen+1):
    for j in range(len(trainloader)):
        latent.data.resize_(args.batchSize, args.nz, 1, 1).copy_(utils.generate_latent_tensor(args.batchSize, args.nz, args.noise_unconnex))
        fake = netGImage(latent).cpu()
        fake.data.resize_(args.batchSize, dim)
        D, I = index.search(fake.data.numpy(), args.k)
        D /= (imageSize ** 2 * 1.0)
        for nbs in range(I.shape[0]):
            for elem in range(I.shape[1]):
                if D[nbs][elem] < args.threshold:
                    el = I[nbs][elem]
                    found_img[el] = found_img.get(el, 0) + 1
        X += [100.0 * len(found_img) / index.ntotal]
        print('[%d/%d] [%d/%d] %d/%d (%.2f)' % (
                i, args.n_gen, j, len(trainloader), len(found_img), index.ntotal, 100.0 * len(found_img) / index.ntotal))

plt.plot(X)
plt.xlabel('Batches')
plt.ylabel('Unique training samples generated')
plt.savefig(args.netG + '.png')
