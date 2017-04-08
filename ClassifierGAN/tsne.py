from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

sys.path.append("..")
import utils
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# For printing in real time on HPC
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

parser = argparse.ArgumentParser()
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--dataset', type=str, default='cifar10',help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../data', type=str, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--imageSize', required=True, type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG',default='', help="path to netG (to create images)")
parser.add_argument('--name', default='dcgan', help='Name of the saved modle')
parser.add_argument('--training-size', type=int, default=10000, help='How many generated samples we train on, -1 = Infinity')
parser.add_argument('--train-real', action='store_true', help='Train classifier on real data or not, useful for knowing the accuracy of classifier')
parser.add_argument('--log-interval', type=int, default=100, help='Number of batchs between prints')
parser.add_argument('--bng-momentum', type=float, default=0.1, help='Momentum of BatchNorm Generator')
parser.add_argument('--model-g', type=str, default='base', help='Generator model : base | upsampling | residual')
args = parser.parse_args()

args.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

###
# Load Generative model, Import Classifier, dataset
###

sys.path.append("../models/ClassifierNN")
sys.path.append("../models/GenerativeNN")

if args.dataset == 'cifar10':
    import ClassifierCifar10 as classifier
    import GenCifar10 as ModelG
    print('No Classifier implemented yet ...')
    nc = 3
elif args.dataset == 'mnist':
    import ClassifierMNIST as classifier
    if args.model_g == 'base':
        import GenMnist as ModelG
    elif args.model_g == 'upsampling':
        import GenMnistUpsampling as ModelG
    elif args.model_g == 'residual':
        import GenMnistResidual as ModelG
    nc = 1
    if args.imageSize != 28:
        print('Model do not work with this image size !')
print(args)

model = classifier.Net()

trainloader, _, n_class = utils.load_dataset(args.dataset, args.dataroot, args.training_size, args.imageSize, args.workers, args.training_size)

if not args.train_real:
    netG = ModelG._netG(args.nz + n_class, args.ngf, nc, args.bng_momentum)
    netG.load_state_dict(torch.load(args.netG))

    print(netG)

    if args.cuda:
        netG.cuda()

    data, target = utils.generate_dataset(netG, args.training_size, args.training_size, args.workers, args.nz, n_class)

if args.train_real:
    iterator = iter(trainloader)
    data, target = iterator.next()

data.resize_(data.size(0), 28 * 28)
print(data.size())
target = target.numpy()

print('Fit tsne')
tsne = TSNE(verbose=2)
Y = tsne.fit_transform(data.numpy())

cmap = plt.cm.get_cmap('jet',max(target)-min(target)+1)


print('After TSNE Word embedding : {}'.format(Y.shape))
plt.scatter(Y.transpose()[0].transpose(), Y.transpose()[1].transpose(), c=target, cmap=cmap)
cb = plt.colorbar(ticks= np.arange(0, 10, 1) + 0.5)

cb.set_ticklabels(np.unique(target))


plt.savefig('tsne.png', dpi=1000)
