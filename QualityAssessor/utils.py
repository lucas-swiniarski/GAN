import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset

###
# custom weights initialization called on netG and netD.
# Allows networks to be efficiently trained.
###

def load_dataset(dataset, dataroot, batchSize, imageSize, workers):
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Scale(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = dset.CIFAR10(root=dataroot, train=True, download=True, transform=transform)
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
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=int(workers))

        testset = dset.MNIST(root=dataroot, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=int(workers))
        n_class = 10 # The number of classes
    return trainloader, testloader, n_class

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

###
# Parallel forward if using more than one gpu.
###

def parallel_forward(model, input, ngpu):
    gpu_ids = None
    if isinstance(input.data, torch.cuda.FloatTensor) and ngpu > 1:
        gpu_ids = range(ngpu)
    return nn.parallel.data_parallel(model, input, gpu_ids)

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

def generate_latent_tensor(batchSize, nz, n_class, target=None):
    if not torch.is_tensor(target):
        target = torch.LongTensor(batchSize).random_(0, n_class)
    return torch.cat((one_hot_encoder(target, n_class), torch.FloatTensor(batchSize, nz).normal_(0, 1)), 1).unsqueeze(2).unsqueeze(3)

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

    tensorDataset = torch.utils.data.TensorDataset(data_tensor, target_tensor)
    loader = torch.utils.data.DataLoader(tensorDataset, batch_size=batchSize, shuffle=True, num_workers=int(workers))
    return loader
