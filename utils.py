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

def load_dataset(dataset, dataroot, batchSize, imageSize, workers, trainsetsize):
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Scale(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = dset.CIFAR10(root=dataroot, train=True, download=True, transform=transform)

        if trainsetsize > 0:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainsetsize, shuffle=True, num_workers=int(workers))
            iterator = iter(trainloader)
            data, labels = iterator.next()
            trainset = torch.utils.data.TensorDataset(data, labels)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=int(workers))

        testset = dset.CIFAR10(root=dataroot, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=int(workers))
        n_class = 10
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Scale(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = dset.MNIST(root=dataroot, train=True, download=True, transform=transform)

        if trainsetsize > 0:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainsetsize, shuffle=True, num_workers=int(workers))
            iterator = iter(trainloader)
            data, labels = iterator.next()
            trainset = torch.utils.data.TensorDataset(data, labels)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=int(workers))

        testset = dset.MNIST(root=dataroot, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=int(workers))
        n_class = 10 # The number of classes
    elif dataset == 'stl10':
        transform = transforms.Compose([
            transforms.Scale(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = dset.STL10(root=dataroot, split='train+unlabeled', download=True, transform=transform)

        if trainsetsize > 0:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainsetsize, shuffle=True, num_workers=int(workers))
            iterator = iter(trainloader)
            data, labels = iterator.next()
            trainset = torch.utils.data.TensorDataset(data, labels)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=int(workers))

        testset = dset.MNIST(root=dataroot, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=int(workers))
        n_class = 10 # The number of classes
    return trainloader, testloader, n_class

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

def weights_clamp(m, c=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.clamp_(-c, c)
        if m.bias is not None:
            m.bias.data.clamp_(-c,c)
    elif classname.find('BatchNorm2d') != -1:
        if m.weight is not None:
            m.weight.data.clamp_(-c, + c)
            m.bias.data.clamp_(-c,c)

###
# One hot encoder : Given a target vector ( batchsize x 1 ) : return matrix (batchsize x n_class )
###

def one_hot_encoder(target, n_class):
    y_one_hot = torch.FloatTensor(target.size(0), n_class)
    y_one_hot.zero_()
    y_one_hot.scatter_(1, target.unsqueeze(1), 1)
    return y_one_hot

###
# Generate a latent vector to feed a generator. If the target is not defined, a random target is created.
###

def generate_latent_tensor(batchSize, nz, noise_unconnex, n_class=0, target=None, info_gan_latent=0):
    if noise_unconnex:
        latent = torch.FloatTensor(batchSize, nz).normal_(0, 0.1)
        latent.add_(torch.from_numpy(np.random.randint(-1, 2, (batchSize, nz))).float())
    else:
        latent = torch.FloatTensor(batchSize, nz).normal_(0, 1)

    if n_class != 0:
        if not torch.is_tensor(target):
            target = torch.LongTensor(batchSize).random_(0, n_class)
        class_one_hot = one_hot_encoder(target, n_class)
        latent = torch.cat((class_one_hot, latent), 1)

    if info_gan_latent != 0:
        latent_code = torch.FloatTensor(batchSize, info_gan_latent).uniform_(-1,1)
        latent = torch.cat((latent_code, latent), 1)
    return latent.unsqueeze(2).unsqueeze(3)

def get_latent_code(latent, info_gan_latent):
    return latent[:,0:info_gan_latent]


###
# Generate a data-set from a generator. Care to give a non-cuda generator !
###

def generate_dataset(netG, size, batchSize, workers, nz, n_class):
    latent = torch.FloatTensor(10, nz + n_class, 1, 1)
    target = torch.LongTensor(10)

    latent = Variable(latent)
    target = Variable(target)

    batch_number = int(size) / 10
    for i in range(batch_number):
        latent.data.copy_(generate_latent_tensor(10, nz, n_class, torch.range(0,9).long()))
        target.data.copy_(torch.range(0,9).long())

        output = netG(latent)
        if i == 0:
            target_tensor = target
            data_tensor = output
        else:
            target_tensor = torch.cat((target_tensor, target), 0)
            data_tensor = torch.cat((data_tensor, output),0)

    return data_tensor.data, target_tensor.data

###
# Concatenate two tensors first dimension and mixing : input1 input2 -> input1[0], input2[0], input1[1]
###

def mix(input1, input2):
    x = torch.randn(2, 3)
    result = torch.cat((input2, input1), 0)
    for i in range(input1.size(0)):
        result[2 * i] = input1[i]
        result[2 * i + 1] = input2[i]
    return result
