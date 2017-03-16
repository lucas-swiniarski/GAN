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
import utils

# For printing in real time on HPC
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../data', type=str, help='path to dataset')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', required=True, type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', required=True,default='', help="path to netG (to create images)")
parser.add_argument('--name', default='dcgan', help='Name of the saved modle')
parser.add_argument('--training-size', type=int, default=-1, help='How many generated samples we train on, -1 = Infinity')

args = parser.parse_args()

args.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

###
# Load Generative model, Import Classifier, dataset
###

sys.path.append("../ClassifierNN")
sys.path.append("../GenerativeNN")

if args.dataset == 'cifar10':
    print('No Classifier implemented yet ...')
    nc = 3
elif args.dataset == 'mnist':
    import ClassifierMNIST as classifier
    import GenMnist as generator
    nc = 1
    if args.imageSize != 28:
        print('Model do not work with this image size !')
print(args)

model = classifier.Net()
if args.cuda:
    model.cuda()

trainloader_data, validloader_data, n_class = utils.load_dataset(args.dataset, args.dataroot, args.batchSize, args.imageSize, args.workers)

assert trainloader_data, validloader_data

netG = generator._netG(args.nz + n_class, args.ngf, nc)
netG = netG.load_state_dict(torch.load(args.netG))
print(netG)

###
# Create Generative validation set and Generative training set if necessary.
###


validloader_gen = utils.generate_dataset(netG, len(validloader_data.dataset), args.nz, args.cuda, n_class)

if args.training_size != -1:
    trainloader_gen = utils.generate_dataset(netG, args.training_size, args.nz, args.cuda, n_class)

optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (args.beta1, 0.999))

def train(epoch, train_loader=None):
    model.train()
    if train_loader != None:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
    else:
        latent = torch.FloatTensor(args.batchSize, args.nz + n_class, 1, 1)
        target = torch.LongTensor(args.batchSize)

        if args.cuda:
            latent, target = latent.cuda(), target.cuda()

        latent = Variable(latent)
        target = Variable(target)

        for batch_idx in range(len(trainloader_data)):
            labels = torch.FloatTensor(args.batchSize).random_(0, 9)
            target.data.copy_(labels)
            latent.data.copy_(utils.generate_latent_tensor(args.batchSize, args.nz, n_class, labels))

            data = netG(latent)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader_data.dataset),
                    100. * batch_idx / len(trainloader_data), loss.data[0]))

def test(epoch, valid_loader, dataset_name):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        dataset_name,
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    if args.training_size == -1:
        train(epoch)
    else:
        train(epoch, trainloader_gen)
    test(epoch, validloader_gen, 'Generative Validation set')
    test(epoch, validloader_data, 'Test set')
