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
    elif args.dataset == 'stl10':
        transform = transforms.Compose([
            transforms.Scale(args.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = dset.STL10(root=args.dataroot, split='train+unlabeled', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, num_workers=int(args.workers))
        nc = 3
    else:
        transform = transforms.Compose([
            transforms.Scale(args.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = dset.ImageFolder(root=args.dataroot + args.dataset, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, num_workers=int(args.workers))
        nc = 3

    return trainloader, nc


###
# Concatenate two tensors first dimension and mixing : input1 input2 -> input1[0], input2[0], input1[1]
# Used to visualize reconstructed images
###

def mix(input1, input2):
    x = torch.randn(2, 3)
    result = torch.cat((input2, input1), 0)
    for i in range(input1.size(0)):
        result[2 * i] = input1[i]
        result[2 * i + 1] = input2[i]
    return result
