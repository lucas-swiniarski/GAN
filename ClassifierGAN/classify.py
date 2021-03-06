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

# For printing in real time on HPC
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../data', type=str, help='path to dataset')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', required=True, type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG',default='', help="path to netG (to create images)")
parser.add_argument('--name', default='dcgan', help='Name of the saved modle')
parser.add_argument('--training-size', type=int, default=-1, help='How many generated samples we train on, -1 = Infinity')
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

trainloader_real, validloader_real, n_class = utils.load_dataset(args.dataset, args.dataroot, args.batchSize, args.imageSize, args.workers, args.training_size)

assert trainloader_real, validloader_real

if not args.train_real:
    netG = ModelG._netG(args.nz + n_class, args.ngf, nc, args.bng_momentum)
    netG.load_state_dict(torch.load(args.netG))
    print(netG)
    if args.cuda:
        netG.cuda()

###
# Create Generative validation set and Generative training set if necessary.
###

# If we need one day a validation set on generated data.
# print('Create Validation set with generated data ...')
# validloader_gen = utils.generate_dataset(netG, 10000, args.batchSize, args.workers, args.nz, n_class)

if (args.training_size != -1) and (args.train_real == False):
    trainloader_gen = utils.generate_dataset(netG, args.training_size, args.batchSize, args.workers, args.nz, n_class)

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
criterion = nn.CrossEntropyLoss()

def train(epoch, train_loader=None):
    model.train()
    if train_loader != None:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
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

        for batch_idx in range(len(trainloader_real)):
            labels = torch.FloatTensor(args.batchSize).random_(0, 9)
            target.data.copy_(labels)
            latent.data.copy_(utils.generate_latent_tensor(args.batchSize, args.nz, n_class, labels.long()))

            data = netG(latent)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader_real.dataset),
                    100. * batch_idx / len(trainloader_real), loss.data[0]))

def test(epoch, valid_loader, dataset_name):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(valid_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        dataset_name,
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    if args.train_real:
        train(epoch, trainloader_real)
    elif args.training_size == -1:
        train(epoch)
    else:
        train(epoch, trainloader_gen)
    # test(epoch, validloader_gen, 'Generative Validation set')
    test(epoch, validloader_real, 'Test set')
