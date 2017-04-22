import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset
import numpy as np

###
# custom weights initialization called on netG and netD.
# Allows networks to be efficiently trained.
###

def load_dataset(args):
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Scale(args.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = dset.CIFAR10(root=args.dataroot, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, num_workers=int(args.workers))
        nc = 3
    elif args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Scale(args.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = dset.MNIST(root=args.dataroot, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, num_workers=int(args.workers))
        nc = 1
    elif dataset == 'stl10':
        transform = transforms.Compose([
            transforms.Scale(args.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = dset.STL10(root=dataroot, split='train+unlabeled', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=int(args.workers))
        nc = 3
    return trainloader, nc
